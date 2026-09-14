#!/usr/bin/env python3
"""Generate print, texture, mesh, and metric manifest assets for the paper pad."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import yaml


MODULES_PER_MARKER = 6  # 4x4 payload + one black border cell on every side
MM_PER_M = 1000.0
POINTS_PER_MM = 72.0 / 25.4


def aruco_dictionary(name):
    if not hasattr(cv2.aruco, name):
        raise ValueError("OpenCV has no ArUco dictionary %s" % name)
    dictionary_id = getattr(cv2.aruco, name)
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        return cv2.aruco.getPredefinedDictionary(dictionary_id)
    return cv2.aruco.Dictionary_get(dictionary_id)


def marker_grid(dictionary, marker_id, border_bits):
    side = 4 + 2 * border_bits
    if side != MODULES_PER_MARKER:
        raise ValueError("this Fig. 1 generator requires one border bit")
    if hasattr(cv2.aruco, "generateImageMarker"):
        image = cv2.aruco.generateImageMarker(dictionary, marker_id, side, borderBits=border_bits)
    else:
        image = cv2.aruco.drawMarker(dictionary, marker_id, side, borderBits=border_bits)
    return image < 128


def load_layout(path):
    raw = Path(path).read_bytes()
    layout = yaml.safe_load(raw)
    canvas = float(layout["canvas_units"])
    markers = layout["markers"]
    ids = [int(item["id"]) for item in markers]
    if not markers or len(set(ids)) != len(markers):
        raise ValueError("pad layout must contain at least one marker with unique IDs")
    for marker in markers:
        x, y, size = (float(marker[key]) for key in ("x", "y", "size"))
        if size <= 0 or x < 0 or y < 0 or x + size > canvas or y + size > canvas:
            raise ValueError("marker %s lies outside the design canvas" % marker["id"])
    return layout, hashlib.sha256(raw).hexdigest()


def black_rectangles(layout):
    dictionary = aruco_dictionary(layout["dictionary"])
    border_bits = int(layout.get("marker_border_bits", 1))
    rectangles = []
    for marker in layout["markers"]:
        x, y, size = (float(marker[key]) for key in ("x", "y", "size"))
        cell = size / MODULES_PER_MARKER
        grid = marker_grid(dictionary, int(marker["id"]), border_bits)
        for row, col in np.argwhere(grid):
            rectangles.append((x + col * cell, y + row * cell, cell, cell))
    return rectangles


def write_svg(path, rectangles, canvas_units, size_m):
    size_mm = size_m * MM_PER_M
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" width="%.6fmm" height="%.6fmm" '
        'viewBox="0 0 %.6f %.6f" shape-rendering="crispEdges">'
        % (size_mm, size_mm, canvas_units, canvas_units),
        '<rect width="100%%" height="100%%" fill="white"/>',
        '<g fill="black" stroke="none">',
    ]
    lines.extend(
        '<rect x="%.9f" y="%.9f" width="%.9f" height="%.9f"/>' % rect
        for rect in rectangles
    )
    lines.extend(["</g>", "</svg>"])
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def pdf_document(page_points, content):
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %.6f %.6f] "
            "/Resources << >> /Contents 4 0 R >>" % (page_points, page_points)
        ).encode("ascii"),
        b"<< /Length %d >>\nstream\n" % len(content) + content + b"endstream",
    ]
    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(("%d 0 obj\n" % index).encode("ascii"))
        output.extend(obj)
        output.extend(b"\nendobj\n")
    xref = len(output)
    output.extend(("xref\n0 %d\n" % (len(objects) + 1)).encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(("%010d 00000 n \n" % offset).encode("ascii"))
    output.extend(
        (
            "trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n"
            % (len(objects) + 1, xref)
        ).encode("ascii")
    )
    return bytes(output)


def write_pdf(path, rectangles, canvas_units, size_m):
    page_points = size_m * MM_PER_M * POINTS_PER_MM
    scale = page_points / canvas_units
    operations = ["q", "0 0 0 rg"]
    for x, y, width, height in rectangles:
        pdf_y = canvas_units - y - height
        operations.append(
            "%.8f %.8f %.8f %.8f re f"
            % (x * scale, pdf_y * scale, width * scale, height * scale)
        )
    operations.append("Q")
    content = ("\n".join(operations) + "\n").encode("ascii")
    Path(path).write_bytes(pdf_document(page_points, content))


def write_png(path, rectangles, canvas_units, texture_px):
    image = np.full((texture_px, texture_px), 255, dtype=np.uint8)
    for x, y, width, height in rectangles:
        x0 = int(round(x / canvas_units * texture_px))
        y0 = int(round(y / canvas_units * texture_px))
        x1 = int(round((x + width) / canvas_units * texture_px))
        y1 = int(round((y + height) / canvas_units * texture_px))
        image[y0:y1, x0:x1] = 0
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 9]):
        raise RuntimeError("failed to write %s" % path)


def metric_manifest(layout, layout_hash, size_m):
    canvas = float(layout["canvas_units"])
    scale = size_m / canvas
    markers = []
    for marker in sorted(layout["markers"], key=lambda item: int(item["id"])):
        marker_id = int(marker["id"])
        x, y, size = (float(marker[key]) for key in ("x", "y", "size"))
        left = (x - canvas / 2.0) * scale
        right = (x + size - canvas / 2.0) * scale
        top = (canvas / 2.0 - y) * scale
        bottom = (canvas / 2.0 - y - size) * scale
        markers.append(
            {
                "id": marker_id,
                "size_m": size * scale,
                "center_m": [(left + right) / 2.0, (top + bottom) / 2.0, 0.0],
                "corners_m": [
                    [left, top, 0.0],
                    [right, top, 0.0],
                    [right, bottom, 0.0],
                    [left, bottom, 0.0],
                ],
            }
        )
    return {
        "format_version": 1,
        "layout_name": layout["name"],
        "layout_sha256": layout_hash,
        "dictionary": layout["dictionary"],
        "pad_size_m": size_m,
        "frame": {
            "origin": "pad center",
            "x": "image right",
            "y": "image up",
            "z": "out of printed face",
        },
        "markers": markers,
    }


def write_unreal_mesh(obj_path, mtl_path, texture_name, size_m):
    half_cm = size_m * 100.0 / 2.0
    obj = """mtllib {mtl}
o landing_pad
v {n:.9f} {n:.9f} 0
v {p:.9f} {n:.9f} 0
v {p:.9f} {p:.9f} 0
v {n:.9f} {p:.9f} 0
vt 0 1
vt 1 1
vt 1 0
vt 0 0
vn 0 0 1
usemtl landing_pad_material
f 1/1/1 2/2/1 3/3/1 4/4/1
""".format(mtl=mtl_path.name, n=-half_cm, p=half_cm)
    mtl = """newmtl landing_pad_material
Ka 1.0 1.0 1.0
Kd 1.0 1.0 1.0
Ks 0.0 0.0 0.0
illum 1
map_Kd {texture}
""".format(texture=texture_name)
    obj_path.write_text(obj, encoding="ascii")
    mtl_path.write_text(mtl, encoding="ascii")


def main():
    package = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--size-m", type=float, required=True, help="physical pad side L")
    parser.add_argument(
        "--layout", default=str(package / "config" / "paper_pad_layout.yaml")
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="paper_pad")
    parser.add_argument("--texture-px", type=int, default=4096)
    args = parser.parse_args()
    if args.size_m <= 0.0:
        parser.error("--size-m must be positive")
    if args.texture_px < 540:
        parser.error("--texture-px must be at least 540")

    layout, layout_hash = load_layout(args.layout)
    canvas = float(layout["canvas_units"])
    rectangles = black_rectangles(layout)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    base = output / args.prefix
    svg_path = base.with_suffix(".svg")
    pdf_path = base.with_suffix(".pdf")
    png_path = base.with_suffix(".png")
    yaml_path = base.with_suffix(".yaml")
    obj_path = base.with_suffix(".obj")
    mtl_path = base.with_suffix(".mtl")

    write_svg(svg_path, rectangles, canvas, args.size_m)
    write_pdf(pdf_path, rectangles, canvas, args.size_m)
    write_png(png_path, rectangles, canvas, args.texture_px)
    yaml_path.write_text(
        yaml.safe_dump(metric_manifest(layout, layout_hash, args.size_m), sort_keys=False),
        encoding="utf-8",
    )
    write_unreal_mesh(obj_path, mtl_path, png_path.name, args.size_m)
    unreal = {
        "mesh_units": "centimeters",
        "mesh_side_cm": args.size_m * 100.0,
        "texture": png_path.name,
        "recommended_texture_settings": {
            "mip_gen_settings": "NoMipmaps",
            "filter": "Nearest",
            "compression": "UserInterface2D",
            "material": "Unlit",
        },
    }
    (output / (args.prefix + "_unreal.json")).write_text(
        json.dumps(unreal, indent=2) + "\n", encoding="utf-8"
    )
    print("generated %d markers at L=%.6fm in %s" % (len(layout["markers"]), args.size_m, output))


if __name__ == "__main__":
    main()
