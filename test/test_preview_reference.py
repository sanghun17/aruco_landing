"""A fixed pad reference survives missing marker heartbeats, not session resets."""
import importlib.util
from pathlib import Path
import threading
import numpy as np
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
import rospy

spec=importlib.util.spec_from_file_location('preview',Path(__file__).parents[1]/'scripts/manual_flight_preview.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class Publisher:
    def __init__(self):self.values=[]
    def publish(self,v):self.values.append(v)

def test_fixed_reference_lifetime(monkeypatch):
    clock=[10.];monkeypatch.setattr(module.rospy,'get_time',lambda:clock[0])
    p=module.Preview.__new__(module.Preview);p.lock=threading.RLock()
    p.pad=None;p.ready=(False,0.);p.poses={};p.body_camera=np.eye(4)
    p.pub={'optitrack':{k:Publisher() for k in ('body','camera','valid')}}
    def pose():
        m=PoseStamped();m.header.frame_id='odom';m.header.stamp=rospy.Time.from_sec(clock[0]);m.pose.orientation.w=1.;return m
    p.pad_pose(pose());p.aligned(Bool(True))
    clock[0]=100.;p.pose(pose(),'optitrack');p.tick(None)
    assert p.pub['optitrack']['valid'].values[-1] is True
    clock[0]+=.3;p.tick(None)
    assert p.pub['optitrack']['valid'].values[-1] is False
    p.pose(pose(),'optitrack');p.aligned(Bool(False));p.tick(None)
    assert p.pad is None and p.pub['optitrack']['valid'].values[-1] is False
    p.pad_pose(pose());p.aligned(Bool(True));clock[0]=1.;p.tick(None)
    assert p.pad is None and not p.ready[0]
