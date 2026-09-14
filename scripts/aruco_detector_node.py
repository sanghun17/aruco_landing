#!/usr/bin/env python3
import math

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Pose, PoseArray, PoseWithCovarianceStamped
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, Int32MultiArray

from aruco_landing.geometry import rotation_matrix_to_quaternion


class ArucoDetectorNode:
    def __init__(self):
        rospy.init_node("aruco_detector")
        if not hasattr(cv2, "aruco"):
            raise RuntimeError("OpenCV was built without the aruco module")

        self.bridge = CvBridge()
        self.target_id = int(rospy.get_param("~target_id", 0))
        self.marker_size = float(rospy.get_param("~marker_size_m", 0.20))
        self.publish_debug = bool(rospy.get_param("~publish_debug_image", True))
        self.translation_stddev = float(rospy.get_param("~translation_stddev_m", 0.05))
        self.rotation_stddev = math.radians(float(rospy.get_param("~rotation_stddev_deg", 5.0)))

        dictionary_name = str(rospy.get_param("~dictionary", "DICT_4X4_100"))
        if not hasattr(cv2.aruco, dictionary_name):
            raise ValueError("unknown ArUco dictionary: %s" % dictionary_name)
        dictionary_id = getattr(cv2.aruco, dictionary_name)
        if hasattr(cv2.aruco, "getPredefinedDictionary"):
            self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        else:
            self.dictionary = cv2.aruco.Dictionary_get(dictionary_id)
        if hasattr(cv2.aruco, "DetectorParameters_create"):
            self.parameters = cv2.aruco.DetectorParameters_create()
        else:
            self.parameters = cv2.aruco.DetectorParameters()
        self.detector = None
        if hasattr(cv2.aruco, "ArucoDetector"):
            self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)

        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_frame = ""
        self.warned_uncalibrated = False

        self.ids_pub = rospy.Publisher("/landing/markers/ids", Int32MultiArray, queue_size=2)
        self.poses_pub = rospy.Publisher("/landing/markers/poses_camera", PoseArray, queue_size=2)
        self.target_pub = rospy.Publisher(
            "/landing/target_pose_camera", PoseWithCovarianceStamped, queue_size=2
        )
        self.visible_pub = rospy.Publisher("/landing/target_visible", Bool, queue_size=2)
        self.debug_pub = rospy.Publisher("/landing/debug/image", Image, queue_size=1)

        image_topic = rospy.get_param("~image_topic", "/landing/camera/image_raw")
        camera_info_topic = rospy.get_param(
            "~camera_info_topic", "/landing/camera/camera_info"
        )
        self.info_sub = rospy.Subscriber(
            camera_info_topic, CameraInfo, self.camera_info_callback, queue_size=1
        )
        self.image_sub = rospy.Subscriber(image_topic, Image, self.image_callback, queue_size=1)
        rospy.loginfo(
            "ArUco detector ready: dictionary=%s target_id=%d marker_size=%.3fm image=%s",
            dictionary_name,
            self.target_id,
            self.marker_size,
            image_topic,
        )

    def camera_info_callback(self, msg):
        matrix = np.asarray(msg.K, dtype=np.float64).reshape(3, 3)
        if msg.width > 0 and msg.height > 0 and matrix[0, 0] > 0.0 and matrix[1, 1] > 0.0:
            self.camera_matrix = matrix
            self.dist_coeffs = np.asarray(msg.D, dtype=np.float64)
            self.camera_frame = msg.header.frame_id
            if self.warned_uncalibrated:
                rospy.loginfo("valid camera calibration received; metric marker pose enabled")
                self.warned_uncalibrated = False

    def detect(self, gray):
        if self.detector is not None:
            return self.detector.detectMarkers(gray)
        return cv2.aruco.detectMarkers(
            gray, self.dictionary, parameters=self.parameters
        )

    @staticmethod
    def make_pose(rvec, tvec):
        rvec = np.asarray(rvec, dtype=np.float64).reshape(3)
        tvec = np.asarray(tvec, dtype=np.float64).reshape(3)
        rotation, _ = cv2.Rodrigues(rvec)
        qx, qy, qz, qw = rotation_matrix_to_quaternion(rotation)
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = map(float, tvec)
        pose.orientation.x = qx
        pose.orientation.y = qy
        pose.orientation.z = qz
        pose.orientation.w = qw
        return pose

    def image_callback(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            rospy.logerr_throttle(2.0, "cv_bridge conversion failed: %s", exc)
            return

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detect(gray)
        ordered_ids = [] if ids is None else [int(x) for x in ids.flatten()]
        self.ids_pub.publish(Int32MultiArray(data=ordered_ids))
        self.visible_pub.publish(Bool(data=self.target_id in ordered_ids))

        rvecs = tvecs = None
        if ordered_ids and self.camera_matrix is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_size, self.camera_matrix, self.dist_coeffs
            )
            poses = PoseArray()
            poses.header = msg.header
            if self.camera_frame:
                poses.header.frame_id = self.camera_frame
            poses.poses = [self.make_pose(rvec, tvec) for rvec, tvec in zip(rvecs, tvecs)]
            self.poses_pub.publish(poses)

            if self.target_id in ordered_ids:
                index = ordered_ids.index(self.target_id)
                target = PoseWithCovarianceStamped()
                target.header = poses.header
                target.pose.pose = poses.poses[index]
                tv = self.translation_stddev ** 2
                rv = self.rotation_stddev ** 2
                target.pose.covariance[0] = tv
                target.pose.covariance[7] = tv
                target.pose.covariance[14] = tv
                target.pose.covariance[21] = rv
                target.pose.covariance[28] = rv
                target.pose.covariance[35] = rv
                self.target_pub.publish(target)
        elif ordered_ids and not self.warned_uncalibrated:
            rospy.logwarn(
                "markers detected but CameraInfo has no valid K; metric poses are withheld"
            )
            self.warned_uncalibrated = True

        if self.publish_debug and self.debug_pub.get_num_connections() > 0:
            debug = image.copy()
            if ordered_ids:
                cv2.aruco.drawDetectedMarkers(debug, corners, ids)
                if rvecs is not None and hasattr(cv2, "drawFrameAxes"):
                    for rvec, tvec in zip(rvecs, tvecs):
                        cv2.drawFrameAxes(
                            debug,
                            self.camera_matrix,
                            self.dist_coeffs,
                            rvec,
                            tvec,
                            self.marker_size * 0.5,
                        )
            out = self.bridge.cv2_to_imgmsg(debug, encoding="bgr8")
            out.header = msg.header
            self.debug_pub.publish(out)


if __name__ == "__main__":
    try:
        ArucoDetectorNode()
        rospy.spin()
    except (rospy.ROSInterruptException, KeyboardInterrupt):
        pass
