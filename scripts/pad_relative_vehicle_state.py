#!/usr/bin/env python3
"""Convert camera-frame pad pose into body pose expressed in the pad frame."""

import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import PoseWithCovarianceStamped
from tf.transformations import quaternion_from_matrix, quaternion_matrix


def transform_matrix(translation, rotation):
    matrix = quaternion_matrix(
        [rotation.x, rotation.y, rotation.z, rotation.w]
    )
    matrix[0:3, 3] = [translation.x, translation.y, translation.z]
    return matrix


class PadRelativeVehicleState:
    def __init__(self):
        rospy.init_node("pad_relative_vehicle_state")
        self.body_frame = rospy.get_param("~body_frame", "base_link")
        self.pad_frame = rospy.get_param("~pad_frame", "landing_pad")
        input_topic = rospy.get_param(
            "~target_pose_topic", "/landing/target_pose_camera"
        )
        output_topic = rospy.get_param(
            "~vehicle_pose_topic", "/landing/vehicle_pose_pad"
        )
        camera_output_topic = rospy.get_param(
            "~camera_pose_topic", "/landing/camera_pose_pad"
        )
        self.buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.listener = tf2_ros.TransformListener(self.buffer)
        self.publisher = rospy.Publisher(
            output_topic, PoseWithCovarianceStamped, queue_size=2
        )
        self.camera_publisher = rospy.Publisher(
            camera_output_topic, PoseWithCovarianceStamped, queue_size=2
        )
        self.subscriber = rospy.Subscriber(
            input_topic,
            PoseWithCovarianceStamped,
            self.pose_callback,
            queue_size=2,
            tcp_nodelay=True,
        )

    def pose_callback(self, message):
        camera_frame = message.header.frame_id
        if not camera_frame:
            rospy.logwarn_throttle(2.0, "target pose has no camera frame; state withheld")
            return
        try:
            camera_from_body = self.buffer.lookup_transform(
                camera_frame,
                self.body_frame,
                message.header.stamp,
                rospy.Duration(0.01),
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as error:
            rospy.logwarn_throttle(
                2.0,
                "camera-to-body TF unavailable (%s <- %s): %s",
                camera_frame,
                self.body_frame,
                error,
            )
            return

        pose = message.pose.pose
        camera_from_pad = transform_matrix(pose.position, pose.orientation)
        pad_from_camera = np.linalg.inv(camera_from_pad)
        tf_message = camera_from_body.transform
        camera_from_body_matrix = transform_matrix(
            tf_message.translation, tf_message.rotation
        )
        pad_from_body = np.matmul(
            pad_from_camera, camera_from_body_matrix
        )
        quaternion = quaternion_from_matrix(pad_from_body)

        output = PoseWithCovarianceStamped()
        output.header.stamp = message.header.stamp
        output.header.frame_id = self.pad_frame
        output.pose.pose.position.x = float(pad_from_body[0, 3])
        output.pose.pose.position.y = float(pad_from_body[1, 3])
        output.pose.pose.position.z = float(pad_from_body[2, 3])
        output.pose.pose.orientation.x = float(quaternion[0])
        output.pose.pose.orientation.y = float(quaternion[1])
        output.pose.pose.orientation.z = float(quaternion[2])
        output.pose.pose.orientation.w = float(quaternion[3])

        # First-order rotation of the translational covariance. Lever-arm and
        # orientation coupling are intentionally not invented here.
        covariance = np.asarray(message.pose.covariance, dtype=float).reshape(6, 6)
        rotation = np.linalg.inv(camera_from_pad)[0:3, 0:3]
        output_covariance = np.zeros((6, 6), dtype=float)
        output_covariance[0:3, 0:3] = np.matmul(
            np.matmul(rotation, covariance[0:3, 0:3]), rotation.T
        )
        output_covariance[3:6, 3:6] = covariance[3:6, 3:6]
        output.pose.covariance = output_covariance.reshape(-1).tolist()
        camera_output = PoseWithCovarianceStamped()
        camera_output.header = output.header
        camera_quaternion = quaternion_from_matrix(pad_from_camera)
        camera_output.pose.pose.position.x = float(pad_from_camera[0, 3])
        camera_output.pose.pose.position.y = float(pad_from_camera[1, 3])
        camera_output.pose.pose.position.z = float(pad_from_camera[2, 3])
        camera_output.pose.pose.orientation.x = float(camera_quaternion[0])
        camera_output.pose.pose.orientation.y = float(camera_quaternion[1])
        camera_output.pose.pose.orientation.z = float(camera_quaternion[2])
        camera_output.pose.pose.orientation.w = float(camera_quaternion[3])
        camera_output.pose.covariance = output_covariance.reshape(-1).tolist()
        self.camera_publisher.publish(camera_output)
        self.publisher.publish(output)


if __name__ == "__main__":
    PadRelativeVehicleState()
    rospy.spin()
