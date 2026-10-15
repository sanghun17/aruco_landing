"""ROS message/worker tests without a master, services, or actuator transport."""
import importlib.util
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch,Mock
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from aruco_landing.landing_trial import Trial
from aruco_landing.pose_alignment import pose_matrix

spec=importlib.util.spec_from_file_location('terminal_trial_node',Path(__file__).resolve().parents[1]/'scripts/landing_trial_node.py')
node=importlib.util.module_from_spec(spec);spec.loader.exec_module(node)

class InlineThread:
    def __init__(self,target,**kw):self.target=target
    def start(self):self.target()

class TerminalNodeTest(unittest.TestCase):
    def setUp(self):
        self.n=node.LandingTrial.__new__(node.LandingTrial)
        n=self.n;n.lock=threading.RLock();n.dry=False;n.cut_pending=False;n.cut_result=None
        n.trial=Trial(phase='CUT_WAIT',finish_mode='force_disarm',termination_height_source='optitrack')
        n.global_frame='odom';n.aligned=True
        # Nonzero pad placement, rotated axes and camera mount exercise the frame conversion.
        n.pad=pose_matrix([1.,2.,.7],[0.,0.,np.sin(.3),np.cos(.3)])
        n.X=pose_matrix([.02,-.01,.18],[0.,0.,0.,1.])
        G=n.pad@pose_matrix([.1,.2,.01],[0.,0.,0.,1.])
        msg=PoseStamped();msg.header.frame_id='odom';msg.header.stamp=rospy.Time.from_sec(100.)
        p,q=node.matrix_pose(G)
        msg.pose.position.x,msg.pose.position.y,msg.pose.position.z=p
        msg.pose.orientation.x,msg.pose.orientation.y,msg.pose.orientation.z,msg.pose.orientation.w=q
        n.inputs={'mocap':msg,'state':SimpleNamespace(mode='OFFBOARD',armed=True)}
        n.receipts={'mocap':10.};n.marker=Mock(return_value=(False,None,None));n.health=Mock(return_value=True)
        n.termination_service=Mock(return_value=SimpleNamespace(success=True))
        self.clock=patch.object(node.rospy,'get_time',return_value=100.);self.clock.start();self.addCleanup(self.clock.stop)

    def test_height_is_camera_relative_to_pad_not_world_or_body_z(self):
        self.assertAlmostEqual(self.n.mocap_camera_height(10.1),.19)

    def test_stale_header_receipt_alignment_and_wrong_frame_withhold_height(self):
        n=self.n
        self.assertIsNone(n.mocap_camera_height(10.21))
        n.inputs['mocap'].header.stamp=rospy.Time.from_sec(99.)
        self.assertIsNone(n.mocap_camera_height(10.1))
        n.inputs['mocap'].header.stamp=rospy.Time.from_sec(100.)
        n.inputs['mocap'].header.frame_id='map'
        self.assertIsNone(n.mocap_camera_height(10.1))
        n.inputs['mocap'].header.frame_id='odom';n.aligned=False
        self.assertIsNone(n.mocap_camera_height(10.1))

    def request(self):
        with patch.object(node.threading,'Thread',InlineThread),patch.object(node.time,'monotonic',return_value=10.1):
            self.n.request_cut()

    def test_worker_calls_only_generic_safety_without_marker(self):
        self.request();self.n.termination_service.assert_called_once_with()
        self.assertTrue(self.n.cut_result)

    def test_worker_rechecks_freshness_and_pilot_takeover(self):
        self.n.receipts['mocap']=9.
        self.request();self.n.termination_service.assert_not_called()
        self.assertFalse(self.n.cut_result)
        self.n.receipts['mocap']=10.;self.n.inputs['state'].mode='POSCTL'
        self.request();self.n.termination_service.assert_not_called()

    def test_worker_does_not_cut_from_failed_hold(self):
        self.n.trial.phase='FAILED_HOLD';self.request()
        self.n.termination_service.assert_not_called()
