#!/usr/bin/env python3
"""Safely route OptiTrack or a globally aligned landing-marker pose to MAVROS."""

from collections import deque
import json
import os
import threading

import numpy as np
import rospy
import yaml
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from std_msgs.msg import Bool, Int32MultiArray, String
from std_srvs.srv import SetBool, SetBoolResponse, Trigger, TriggerResponse

from aruco_landing.pose_alignment import (
    PadAlignmentEstimator,
    aligned_global_body,
    matrix_pose,
    pose_matrix,
    quaternion_distance_deg,
)


def message_matrix(message):
    pose = message.pose.pose if hasattr(message.pose, "pose") else message.pose
    return pose_matrix(
        (pose.position.x, pose.position.y, pose.position.z),
        (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w),
    )


def message_stamp(message):
    stamp = message.header.stamp
    return stamp if not stamp.is_zero() else rospy.Time.now()


class LandingVisionPoseAdapter:
    """Keep one MAVROS input stable while its upstream estimator changes."""

    def __init__(self):
        rospy.init_node("landing_vision_pose_adapter")
        self.lock = threading.RLock()
        self.global_frame = rospy.get_param("~global_frame", "odom")
        self.mocap_topic = rospy.get_param(
            "~mocap_topic", "/vrpn_client_node/pure/pose"
        )
        self.marker_topic = rospy.get_param(
            "~marker_pose_topic", "/landing/vehicle_pose_pad"
        )
        self.output_topic = rospy.get_param(
            "~output_topic", "/mavros/vision_pose/pose"
        )
        self.allow_marker_switch = bool(rospy.get_param(
            "~allow_marker_switch", False
        ))
        self.auto_switch = bool(rospy.get_param("~auto_switch", False))
        self.max_pair_dt_s = float(rospy.get_param("~max_pair_dt_s", 0.035))
        self.max_source_age_s = float(rospy.get_param("~max_source_age_s", 0.20))
        self.min_marker_inliers = int(rospy.get_param("~min_marker_inliers", 3))
        self.max_switch_translation_jump_m = float(rospy.get_param(
            "~max_switch_translation_jump_m", 0.15
        ))
        self.max_switch_rotation_jump_deg = float(rospy.get_param(
            "~max_switch_rotation_jump_deg", 12.0
        ))
        self.alignment_file = rospy.get_param("~alignment_file", "")

        self.estimator = PadAlignmentEstimator(
            min_samples=int(rospy.get_param("~alignment_min_samples", 30)),
            window_size=int(rospy.get_param("~alignment_window_size", 240)),
            translation_outlier_m=float(rospy.get_param(
                "~alignment_translation_outlier_m", 0.08
            )),
            rotation_outlier_deg=float(rospy.get_param(
                "~alignment_rotation_outlier_deg", 10.0
            )),
            max_translation_std_m=float(rospy.get_param(
                "~alignment_max_translation_std_m", 0.03
            )),
            max_rotation_std_deg=float(rospy.get_param(
                "~alignment_max_rotation_std_deg", 5.0
            )),
        )
        self.source = "optitrack"
        self.visible = False
        self.latest_inlier_count = 0
        self.mocap_buffer = deque(maxlen=300)
        self.latest_mocap = None
        self.latest_marker = None
        self.latest_marker_global = None
        self.last_switch_check = None

        self.output_publisher = rospy.Publisher(
            self.output_topic, PoseStamped, queue_size=2
        )
        self.marker_publisher = rospy.Publisher(
            "/landing/vision_pose_marker", PoseStamped, queue_size=2
        )
        self.pad_publisher = rospy.Publisher(
            "/landing/pad_pose_global", PoseStamped, queue_size=1, latch=True
        )
        self.source_publisher = rospy.Publisher(
            "/landing/vision_pose_source", String, queue_size=1, latch=True
        )
        self.ready_publisher = rospy.Publisher(
            "/landing/pose_transition/ready", Bool, queue_size=1, latch=True
        )
        self.status_publisher = rospy.Publisher(
            "/landing/pose_transition/status", String, queue_size=2, latch=True
        )

        rospy.Subscriber(
            self.mocap_topic, PoseStamped, self.mocap_callback,
            queue_size=100, tcp_nodelay=True,
        )
        rospy.Subscriber(
            self.marker_topic, PoseWithCovarianceStamped, self.marker_callback,
            queue_size=20, tcp_nodelay=True,
        )
        rospy.Subscriber(
            "/landing/target_visible", Bool, self.visible_callback, queue_size=20
        )
        rospy.Subscriber(
            "/landing/estimator/inlier_ids", Int32MultiArray,
            self.inlier_callback, queue_size=20,
        )
        rospy.Service(
            "/landing/pose_transition/select_marker", SetBool,
            self.select_marker_service,
        )
        rospy.Service(
            "/landing/pose_transition/reset_alignment", Trigger,
            self.reset_alignment_service,
        )
        rospy.Service(
            "/landing/pose_transition/save_alignment", Trigger,
            self.save_alignment_service,
        )
        rospy.Timer(rospy.Duration(0.1), self.status_timer)

        if self.alignment_file and os.path.isfile(self.alignment_file):
            self.load_alignment(self.alignment_file)
        self.source_publisher.publish(String(data=self.source))
        rospy.loginfo(
            "landing vision-pose adapter: %s -> %s; marker switching %s",
            self.mocap_topic,
            self.output_topic,
            "enabled" if self.allow_marker_switch else "locked out",
        )

    def publish_matrix(self, publisher, transform, stamp):
        translation, quaternion = matrix_pose(transform)
        message = PoseStamped()
        message.header.stamp = stamp
        message.header.frame_id = self.global_frame
        message.pose.position.x = float(translation[0])
        message.pose.position.y = float(translation[1])
        message.pose.position.z = float(translation[2])
        message.pose.orientation.x = float(quaternion[0])
        message.pose.orientation.y = float(quaternion[1])
        message.pose.orientation.z = float(quaternion[2])
        message.pose.orientation.w = float(quaternion[3])
        publisher.publish(message)

    def mocap_callback(self, message):
        stamp = message_stamp(message)
        transform = message_matrix(message)
        receipt = rospy.get_time()
        with self.lock:
            sample = (stamp.to_sec(), transform, receipt)
            self.mocap_buffer.append(sample)
            self.latest_mocap = sample
            if self.source == "optitrack":
                self.publish_matrix(self.output_publisher, transform, stamp)

    def visible_callback(self, message):
        with self.lock:
            self.visible = bool(message.data)

    def inlier_callback(self, message):
        with self.lock:
            self.latest_inlier_count = len(message.data)

    def nearest_mocap(self, stamp_seconds):
        if not self.mocap_buffer:
            return None
        sample = min(
            self.mocap_buffer, key=lambda value: abs(value[0] - stamp_seconds)
        )
        return sample if abs(sample[0] - stamp_seconds) <= self.max_pair_dt_s else None

    def marker_callback(self, message):
        stamp = message_stamp(message)
        pad_from_body = message_matrix(message)
        receipt = rospy.get_time()
        with self.lock:
            self.latest_marker = (stamp.to_sec(), pad_from_body, receipt)
            estimate = self.estimator.estimate()
            quality_valid = (
                self.visible and self.latest_inlier_count >= self.min_marker_inliers
            )
            if quality_valid and not self.estimator.frozen:
                mocap = self.nearest_mocap(stamp.to_sec())
                if mocap is not None:
                    estimate = self.estimator.add(mocap[1], pad_from_body)
            if estimate is None or not estimate["ready"]:
                return

            marker_global = aligned_global_body(
                estimate["transform"], pad_from_body
            )
            self.latest_marker_global = (stamp.to_sec(), marker_global, receipt)
            self.publish_matrix(self.marker_publisher, marker_global, stamp)
            self.publish_matrix(
                self.pad_publisher, estimate["transform"], rospy.Time.now()
            )
            if self.source == "marker" and quality_valid:
                self.publish_matrix(self.output_publisher, marker_global, stamp)

    def source_is_fresh(self, sample):
        return sample is not None and (
            rospy.get_time() - sample[2]
        ) <= self.max_source_age_s

    def switch_check(self):
        estimate = self.estimator.estimate()
        result = {
            "allowed": self.allow_marker_switch,
            "alignment_ready": bool(estimate and estimate["ready"]),
            "marker_fresh": self.source_is_fresh(self.latest_marker_global),
            "mocap_fresh": self.source_is_fresh(self.latest_mocap),
            "translation_jump_m": None,
            "rotation_jump_deg": None,
            "ready": False,
        }
        if not all((
                result["allowed"], result["alignment_ready"],
                result["marker_fresh"], result["mocap_fresh"])):
            return result
        mocap = self.latest_mocap[1]
        marker = self.latest_marker_global[1]
        result["translation_jump_m"] = float(np.linalg.norm(
            mocap[0:3, 3] - marker[0:3, 3]
        ))
        result["rotation_jump_deg"] = quaternion_distance_deg(
            matrix_pose(mocap)[1], matrix_pose(marker)[1]
        )
        result["ready"] = bool(
            result["translation_jump_m"] <= self.max_switch_translation_jump_m
            and result["rotation_jump_deg"] <= self.max_switch_rotation_jump_deg
        )
        return result

    def select_marker_service(self, request):
        with self.lock:
            if not request.data:
                self.source = "optitrack"
                self.source_publisher.publish(String(data=self.source))
                return SetBoolResponse(True, "vision pose source is optitrack")

            check = self.switch_check()
            self.last_switch_check = check
            if not check["ready"]:
                return SetBoolResponse(
                    False, "marker transition rejected: " + json.dumps(check)
                )
            if not self.estimator.frozen:
                self.estimator.freeze()
            self.source = "marker"
            self.source_publisher.publish(String(data=self.source))
            rospy.logwarn(
                "vision pose transitioned optitrack -> marker: %.4fm, %.3fdeg",
                check["translation_jump_m"], check["rotation_jump_deg"],
            )
            return SetBoolResponse(True, "vision pose source is marker")

    def reset_alignment_service(self, _request):
        with self.lock:
            if self.source != "optitrack":
                return TriggerResponse(
                    False, "return to optitrack before resetting alignment"
                )
            self.estimator.clear()
            self.latest_marker_global = None
            return TriggerResponse(True, "alignment window cleared")

    def alignment_document(self):
        estimate = self.estimator.estimate()
        if estimate is None or not estimate["ready"]:
            raise ValueError("alignment is not ready")
        translation, quaternion = matrix_pose(estimate["transform"])
        return {
            "schema_version": 1,
            "global_frame": self.global_frame,
            "pad_frame": "landing_pad",
            "global_from_pad": {
                "translation_m": [float(value) for value in translation],
                "quaternion_xyzw": [float(value) for value in quaternion],
            },
            "quality": {
                key: estimate.get(key) for key in (
                    "sample_count", "inlier_count", "translation_std_m",
                    "translation_p95_m", "rotation_std_deg", "rotation_p95_deg",
                )
            },
        }

    def save_alignment_service(self, _request):
        with self.lock:
            if not self.alignment_file:
                return TriggerResponse(False, "~alignment_file is empty")
            try:
                document = self.alignment_document()
                directory = os.path.dirname(self.alignment_file)
                if directory:
                    os.makedirs(directory, exist_ok=True)
                temporary = self.alignment_file + ".tmp"
                with open(temporary, "w", encoding="utf-8") as stream:
                    yaml.safe_dump(document, stream, sort_keys=False)
                os.replace(temporary, self.alignment_file)
            except (OSError, ValueError, yaml.YAMLError) as error:
                return TriggerResponse(False, str(error))
            return TriggerResponse(True, self.alignment_file)

    def load_alignment(self, path):
        with open(path, "r", encoding="utf-8") as stream:
            document = yaml.safe_load(stream)
        pose = document["global_from_pad"]
        transform = pose_matrix(
            pose["translation_m"], pose["quaternion_xyzw"]
        )
        self.estimator.set_frozen_transform(
            transform, metadata={"path": path, "quality": document.get("quality")}
        )
        rospy.loginfo("loaded frozen landing-pad alignment: %s", path)

    def status_timer(self, _event):
        with self.lock:
            check = self.switch_check()
            estimate = self.estimator.estimate()
            status = {
                "source": self.source,
                "output_topic": self.output_topic,
                "alignment": None if estimate is None else {
                    key: estimate.get(key) for key in (
                        "sample_count", "inlier_count", "translation_std_m",
                        "translation_p95_m", "rotation_std_deg",
                        "rotation_p95_deg", "ready", "frozen", "loaded",
                    ) if key in estimate
                },
                "switch_check": check,
                "last_switch_check": self.last_switch_check,
            }
            self.ready_publisher.publish(Bool(data=check["ready"]))
            self.status_publisher.publish(String(
                data=json.dumps(status, sort_keys=True)
            ))
            if self.auto_switch and self.source == "optitrack" and check["ready"]:
                response = self.select_marker_service(type(
                    "Request", (), {"data": True}
                )())
                if not response.success:
                    rospy.logwarn_throttle(1.0, response.message)


if __name__ == "__main__":
    LandingVisionPoseAdapter()
    rospy.spin()
