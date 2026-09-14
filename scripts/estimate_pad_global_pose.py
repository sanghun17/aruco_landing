#!/usr/bin/env python3
"""Estimate global-from-pad registration from an OptiTrack landing rosbag."""

import argparse
from bisect import bisect_left
import os

import rosbag
import yaml

from aruco_landing.pose_alignment import (
    PadAlignmentEstimator,
    matrix_pose,
    pose_matrix,
)


def message_matrix(message):
    pose = message.pose.pose if hasattr(message.pose, "pose") else message.pose
    return pose_matrix(
        (pose.position.x, pose.position.y, pose.position.z),
        (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w),
    )


def message_time(message, bag_time):
    return (
        message.header.stamp.to_sec()
        if not message.header.stamp.is_zero()
        else bag_time.to_sec()
    )


def nearest_sample(samples, stamps, target, tolerance):
    index = bisect_left(stamps, target)
    candidates = []
    if index < len(samples):
        candidates.append(samples[index])
    if index:
        candidates.append(samples[index - 1])
    if not candidates:
        return None
    best = min(candidates, key=lambda value: abs(value[0] - target))
    return best if abs(best[0] - target) <= tolerance else None


def atomic_yaml(path, document):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        yaml.safe_dump(document, stream, sort_keys=False)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bag")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mocap-topic", default="/vrpn_client_node/pure/pose"
    )
    parser.add_argument(
        "--marker-topic", default="/landing/vehicle_pose_pad"
    )
    parser.add_argument("--global-frame", default="odom")
    parser.add_argument("--pad-frame", default="landing_pad")
    parser.add_argument("--max-pair-dt", type=float, default=0.035)
    parser.add_argument("--min-samples", type=int, default=30)
    parser.add_argument("--window-size", type=int, default=100000)
    args = parser.parse_args()

    mocap_samples = []
    marker_samples = []
    with rosbag.Bag(args.bag) as bag:
        for topic, message, bag_time in bag.read_messages(
                topics=[args.mocap_topic, args.marker_topic]):
            sample = (message_time(message, bag_time), message_matrix(message))
            if topic == args.mocap_topic:
                mocap_samples.append(sample)
            else:
                marker_samples.append(sample)
    mocap_samples.sort(key=lambda value: value[0])
    marker_samples.sort(key=lambda value: value[0])
    mocap_stamps = [value[0] for value in mocap_samples]
    if not mocap_samples or not marker_samples:
        raise RuntimeError(
            "bag needs both %s and %s" % (args.mocap_topic, args.marker_topic)
        )

    estimator = PadAlignmentEstimator(
        min_samples=args.min_samples,
        window_size=max(args.window_size, args.min_samples),
    )
    matched = 0
    for marker_stamp, pad_from_body in marker_samples:
        mocap = nearest_sample(
            mocap_samples, mocap_stamps, marker_stamp, args.max_pair_dt
        )
        if mocap is not None:
            estimator.add(mocap[1], pad_from_body)
            matched += 1
    estimate = estimator.estimate()
    if estimate is None or not estimate["ready"]:
        raise RuntimeError(
            "alignment not qualified: matched=%d estimate=%r" % (matched, estimate)
        )

    translation, quaternion = matrix_pose(estimate["transform"])
    document = {
        "schema_version": 1,
        "global_frame": args.global_frame,
        "pad_frame": args.pad_frame,
        "global_from_pad": {
            "translation_m": [float(value) for value in translation],
            "quaternion_xyzw": [float(value) for value in quaternion],
        },
        "quality": {
            "matched_sample_count": matched,
            "sample_count": estimate["sample_count"],
            "inlier_count": estimate["inlier_count"],
            "translation_std_m": estimate["translation_std_m"],
            "translation_p95_m": estimate["translation_p95_m"],
            "rotation_std_deg": estimate["rotation_std_deg"],
            "rotation_p95_deg": estimate["rotation_p95_deg"],
        },
        "source": {
            "bag": os.path.abspath(args.bag),
            "mocap_topic": args.mocap_topic,
            "marker_topic": args.marker_topic,
            "max_pair_dt_s": args.max_pair_dt,
        },
    }
    atomic_yaml(args.output, document)
    print(yaml.safe_dump(document, sort_keys=False))
    print("alignment: " + os.path.abspath(args.output))


if __name__ == "__main__":
    main()
