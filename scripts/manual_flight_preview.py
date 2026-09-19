#!/usr/bin/env python3
"""Express OptiTrack and candidate body/camera poses in the live pad frame.

Observation only: no MAVROS publishers, mode/arming calls, or persisted pad pose.
"""
import threading
import numpy as np
import rospy
import yaml
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from std_msgs.msg import Bool
from aruco_landing.pose_alignment import pose_matrix, matrix_pose


def transform(message):
    p, q = message.pose.position, message.pose.orientation
    return pose_matrix([p.x, p.y, p.z], [q.x, q.y, q.z, q.w])


class Preview:
    def __init__(self):
        rospy.init_node('manual_flight_preview')
        self.lock = threading.RLock()
        self.pad = None
        self.ready = (False, 0.)
        self.poses = {}
        path = rospy.get_param('~extrinsic_file')
        with open(path) as stream:
            self.body_camera = np.array(yaml.safe_load(stream)['matrix_row_major']).reshape(4, 4)
        self.pub = {}
        for source, topic in [('optitrack', '/vrpn_client_node/pure/pose'),
                              ('transition', '/landing/vision_pose_selected')]:
            base = '/landing/shadow/' + source
            self.pub[source] = {
                'body': rospy.Publisher(base+'/vehicle_pose_pad', PoseWithCovarianceStamped, queue_size=2),
                'camera': rospy.Publisher(base+'/camera_pose_pad', PoseWithCovarianceStamped, queue_size=2),
                'valid': rospy.Publisher(base+'/valid', Bool, queue_size=1),
            }
            rospy.Subscriber(topic, PoseStamped, self.pose, source, queue_size=2, tcp_nodelay=True)
        rospy.Subscriber('/landing/pad_pose_global', PoseStamped, self.pad_pose, queue_size=1)
        rospy.Subscriber('/landing/alignment/ready', Bool, self.aligned, queue_size=1)
        rospy.Timer(rospy.Duration(1./60.), self.tick)

    def aligned(self, message):
        with self.lock:
            self.ready = (message.data, rospy.get_time())
            if not message.data:
                self.pad = None  # explicit session reset invalidates the fixed reference

    def pad_pose(self, message):
        with self.lock:
            self.pad = (message, rospy.get_time())

    def pose(self, message, source):
        with self.lock:
            self.poses[source] = (message, rospy.get_time())

    @staticmethod
    def output(matrix, stamp):
        msg = PoseWithCovarianceStamped()
        msg.header.stamp, msg.header.frame_id = stamp, 'landing_pad'
        p, q = matrix_pose(matrix)
        msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z = p
        (msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
         msg.pose.pose.orientation.z, msg.pose.pose.orientation.w) = q
        # These are preview coordinates; zero covariance is not a calibrated
        # uncertainty estimate and this topic must not feed an EKF.
        return msg

    def tick(self, _event):
        with self.lock:
            now = rospy.get_time()
            # A frozen session transform has no heartbeat expiry. Only moving
            # vehicle samples expire. Explicit ready=false or a clock rewind
            # invalidates the session reference.
            if self.pad is not None and (now < self.pad[1] or now < self.ready[1]):
                self.pad = None
                self.ready = (False, now)
            pad_ok = self.pad is not None and self.ready[0]
            for source, publishers in self.pub.items():
                sample = self.poses.get(source)
                valid = bool(pad_ok and sample and 0 <= now-sample[1] < .2
                             and -.05 <= now-sample[0].header.stamp.to_sec() < .2
                             and sample[0].header.frame_id == self.pad[0].header.frame_id)
                if valid:
                    body = np.linalg.inv(transform(self.pad[0])) @ transform(sample[0])
                    valid = bool(np.isfinite(body).all())
                    if valid:
                        publishers['body'].publish(self.output(body, sample[0].header.stamp))
                        publishers['camera'].publish(self.output(body @ self.body_camera, sample[0].header.stamp))
                publishers['valid'].publish(valid)


if __name__ == '__main__':
    Preview()
    rospy.spin()
