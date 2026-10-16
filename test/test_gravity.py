import unittest
import cv2
import numpy as np
from aruco_landing.gravity import planar_pose_with_normal, AttitudeHistory, GravityReference
from aruco_landing.physical_pad import PhysicalPadDetector, SessionAlignment


def rotation(v):return cv2.Rodrigues(np.asarray(v,dtype=float))[0]

class GravityTest(unittest.TestCase):
    def setUp(self):
        self.K=np.array([[684.,0,388.],[0,686.,353.],[0,0,1.]])
        self.D=np.array([-.348,.139,.00094,-.00045,-.0276])
        self.manifest={'dictionary':'DICT_4X4_100','markers':[{'id':1,'side_m':.0574,'yaw_deg':-90.,'center_m':{'x':.006,'y':-.004}}]}
        self.detector=PhysicalPadDetector(self.manifest)

    def test_distorted_single_marker_metric_pose_and_visual_yaw(self):
        for yaw in [-.8,0.,1.2]:
            R=rotation([3.,.1,.05])@rotation([0.,0.,yaw]);t=np.array([.02,.03,1.8])
            points=self.detector.models[1]
            pix=cv2.projectPoints(points,cv2.Rodrigues(R)[0],t,self.K,self.D)[0]
            T=planar_pose_with_normal(points,pix,self.K,self.D,R[:,2])
            np.testing.assert_allclose(T[:3,:3],R,atol=1e-7)
            np.testing.assert_allclose(T[:3,3],t,atol=1e-7)
            result=self.detector.estimate([pix],[1],self.K,self.D,R[:,2])
            self.assertIsNotNone(result)
            np.testing.assert_allclose(result['camera_from_pad'],T,atol=1e-7)
            self.assertIsNone(self.detector.estimate([pix],[1],self.K,self.D,-R[:,2]))

    def test_normal_changes_do_not_silently_return_unconstrained_pnp(self):
        R=rotation([3.,.1,.05]);t=np.array([.02,.03,1.])
        points=self.detector.models[1];pix=cv2.projectPoints(points,cv2.Rodrigues(R)[0],t,self.K,self.D)[0]
        normal=rotation([.015,-.01,0])@R[:,2]
        out=self.detector.estimate([pix],[1],self.K,self.D,normal)
        self.assertIsNotNone(out)
        np.testing.assert_allclose(out['camera_from_pad'][:3,2],normal,atol=1e-8)

    def test_imu_history_rejects_gaps_extrapolation_and_attitude_jumps(self):
        h=AttitudeHistory();self.assertTrue(h.add(1.,np.eye(3)));self.assertTrue(h.add(1.02,rotation([.02,0,0])))
        np.testing.assert_allclose(h.interpolate(1.01),rotation([.01,0,0]),atol=1e-7)
        self.assertIsNone(h.interpolate(.9));self.assertIsNone(h.interpolate(1.03))
        self.assertFalse(h.add(1.02,np.eye(3)));self.assertFalse(h.add(1.03,np.zeros((3,3))))
        h.add(1.1,np.eye(3));self.assertIsNone(h.interpolate(1.05))
        h.clear();h.add(2.,np.eye(3));h.add(2.01,rotation([1.,0,0]));self.assertIsNone(h.interpolate(2.005))

    def test_reference_learns_fixed_tilt_and_freezes_without_current_gt(self):
        ref=GravityReference(min_samples=20,min_duration_s=.5)
        true=rotation([.003,-.005,.4]);imu=true@rotation([.02,-.01,0])
        for t in np.arange(0,1.,.02):ref.observe(t,true,imu)
        self.assertTrue(ref.ready)
        np.testing.assert_allclose(ref.rotation@imu.T@np.array([0,0,1]),true.T@np.array([0,0,1]),atol=1e-7)
        frozen=ref.rotation.copy();ref.observe(10.,rotation([.2,0,0]),np.eye(3));np.testing.assert_array_equal(ref.rotation,frozen)
        ref.reset();self.assertFalse(ref.ready)
        for t in np.arange(0,1.,.02):ref.observe(t,np.eye(3),rotation([.2,0,0]))
        self.assertFalse(ref.ready)

    def test_level_alignment_preserves_positions_without_absorbing_body_tilt(self):
        s=SessionAlignment(min_samples=5,min_duration_s=.2,window_size=30,level_pad=True)
        yaw=.3;Y=np.eye(4);Y[:3,:3]=rotation([0,0,yaw]);Y[:3,3]=[.2,-.1,.04]
        body=np.eye(4);body[:3,3]=[.1,.02,.3]
        measured=Y@body;measured[:3,:3]=measured[:3,:3]@rotation([.02,-.01,0])
        for t in np.arange(0,1.,.01):s.add_mocap(t,measured)
        for t in np.arange(.02,.8,.02):s.observe(t,body)
        self.assertTrue(s.ready);np.testing.assert_allclose(s.transform,Y,atol=1e-7)
        high=body.copy();high[2,3]=2.
        np.testing.assert_allclose(s.observe(5.,high)[:3,3],(Y@high)[:3,3],atol=1e-7)

if __name__=='__main__':unittest.main()
