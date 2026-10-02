#!/usr/bin/env python3
"""Synthetic correctness/transfer/latency gate for the experimental CUDA detector.

This benchmark does not include Isaac rendering. It cannot justify switching
the evaluation default until equivalent tests pass on actual rendered images.
"""
import argparse
import json
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from aruco_landing.gpu_aruco import GpuArucoDetector, build_library
from aruco_landing.physical_pad import PhysicalPadDetector, inverse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library',type=Path,required=True)
    parser.add_argument('--nvcc')
    parser.add_argument('--build',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pad-manifest',type=Path,help='also check the full calibrated pad and reference PnP')
    args = parser.parse_args()
    if args.build: build_library(args.library,args.nvcc)
    dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    gpu=GpuArucoDetector(dictionary,str(args.library))
    params=cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod=cv2.aruco.CORNER_REFINE_SUBPIX
    cpu=cv2.aruco.ArucoDetector(dictionary,params)
    cv2.setNumThreads(1)
    images=[]
    # Covers IDs, marker rotation, perspective, blur and two contrasts.
    rng=np.random.default_rng(1701)
    for index in range(48):
        image=np.full((720,720),255,np.uint8)
        marker=cv2.aruco.generateImageMarker(dictionary,index%100,120)
        marker=np.rot90(marker,index%4).copy()
        corners=np.array([[260,260],[460,260],[460,460],[260,460]],np.float32)
        corners+=rng.uniform(-30,30,(4,2)).astype(np.float32)
        homography=cv2.getPerspectiveTransform(np.array([[0,0],[119,0],[119,119],[0,119]],np.float32),corners)
        image=cv2.warpPerspective(marker,homography,(720,720),borderValue=255)
        if index%3==0: image=cv2.GaussianBlur(image,(3,3),.7)
        if index%5==0: image=(image.astype(np.float32)*.65+40).astype(np.uint8)
        images.append(image)
    estimator=None
    K=np.array([[708.077114,0.,360.],[0.,708.077114,360.],[0.,0.,1.]])
    if args.pad_manifest:
        import yaml
        estimator=PhysicalPadDetector(yaml.safe_load(args.pad_manifest.read_text()))
        cpu=estimator.detector
        images=[]
        for index in range(24):
            image=np.full((720,720),255,np.uint8)
            rotation=np.array([np.pi,0.10*np.sin(index),0.10*np.cos(index)])
            translation=np.array([.10*np.sin(index),.10*np.cos(index),.7+(index%4)*.35])
            for mid,model in estimator.models.items():
                corners=cv2.projectPoints(model,rotation,translation,K,np.zeros(5))[0].reshape(4,2).astype(np.float32)
                marker=cv2.aruco.generateImageMarker(dictionary,mid,120)
                H=cv2.getPerspectiveTransform(np.array([[0,0],[119,0],[119,119],[0,119]],np.float32),corners)
                image=np.minimum(image,cv2.warpPerspective(marker,H,(720,720),borderValue=255))
            if index%3==0: image=cv2.GaussianBlur(image,(3,3),.7)
            images.append(image)
    gray=np.array(images)
    rgb=torch.from_numpy(np.repeat(gray[...,None],3,axis=-1)).cuda()
    for _ in range(2): gpu.detect(rgb[:1])
    report=dict(scope='synthetic images only; excludes Isaac rendering', images=len(images),
                backend='experimental fixed-threshold CUDA CCL/quad/TLS/4x4 decode', cases=[])
    for batch_size in (1,4,16):
        begin=time.perf_counter(); transferred=0; detections=[]
        for base in range(0,len(images),batch_size):
            rows,bytes_count=gpu.detect(rgb[base:base+batch_size]);transferred+=bytes_count;detections.extend(rows)
        elapsed=time.perf_counter()-begin
        cpu_begin=time.perf_counter(); references=[]
        for image in images:
            corners,ids,_=cpu.detectMarkers(image)
            references.append((corners,[] if ids is None else ids.flatten().tolist()))
        cpu_elapsed=time.perf_counter()-cpu_begin
        tp=fp=fn=0;errors=[];pose_errors=[];pose_dropouts=0
        for (corners,ids),(ref_corners,ref_ids) in zip(detections,references):
            found={mid:corner.reshape(4,2) for mid,corner in zip(ids,corners)}
            reference={mid:corner.reshape(4,2) for mid,corner in zip(ref_ids,ref_corners)}
            # PnP ignores average marker sides shorter than 12 pixels.
            if estimator:
                reference={mid:corner for mid,corner in reference.items()
                           if np.linalg.norm(corner-np.roll(corner,-1,axis=0),axis=1).mean()>=12}
                found={mid:corner for mid,corner in found.items()
                       if np.linalg.norm(corner-np.roll(corner,-1,axis=0),axis=1).mean()>=12}
                a=estimator.estimate(list(reference.values()),list(reference),K,np.zeros(5))
                b=estimator.estimate(list(found.values()),list(found),K,np.zeros(5))
                if a is not None and b is None: pose_dropouts+=1
                if a is not None and b is not None:
                    pose_errors.append(float(np.linalg.norm(inverse(a['camera_from_pad'])[:3,3]-inverse(b['camera_from_pad'])[:3,3])))
            tp+=len(set(found)&set(reference));fp+=len(set(found)-set(reference));fn+=len(set(reference)-set(found))
            for mid in set(found)&set(reference): errors.extend(np.linalg.norm(found[mid]-reference[mid],axis=1).tolist())
        report['cases'].append(dict(batch_size=batch_size,gpu_wall_s=elapsed,cpu_wall_s=cpu_elapsed,
            gpu_images_per_s=len(images)/elapsed,cpu_images_per_s=len(images)/cpu_elapsed,
            grayscale_reference_bytes=gray.nbytes,gpu_transferred_bytes=transferred,
            reference_matches=tp,extra_vs_reference=fp,missed_vs_reference=fn,
            corner_rmse_px=None if not errors else float(np.sqrt(np.mean(np.square(errors)))),
            pnp_dropouts=pose_dropouts,pose_translation_max_delta_m=max(pose_errors,default=None)))
    report['accuracy_gate_passed']=all(r['extra_vs_reference']==0 and r['missed_vs_reference']==0 and
        r['reference_matches']>=len(images) and r['corner_rmse_px'] is not None and r['corner_rmse_px']<.75 and
        r['pnp_dropouts']==0 and (r['pose_translation_max_delta_m'] is None or r['pose_translation_max_delta_m']<.02)
        for r in report['cases'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    gpu.close()
    if not report['accuracy_gate_passed']: raise SystemExit('synthetic accuracy gate failed; keep CPU default')


if __name__=='__main__': main()
