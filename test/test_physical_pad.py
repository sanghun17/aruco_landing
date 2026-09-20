#!/usr/bin/env python3
import unittest
import numpy as np
import cv2
from aruco_landing.physical_pad import PhysicalPadDetector, SessionAlignment, inverse, relative_covariance, center_crop
from aruco_landing.pose_alignment import pose_matrix


def pose(p, yaw=0):
    return pose_matrix(p,[0,0,np.sin(yaw/2),np.cos(yaw/2)])


class PhysicalPadTest(unittest.TestCase):
    def test_center_crop_preserves_distorted_pnp_geometry(self):
        manifest={'dictionary':'DICT_7X7_50','markers':[{'id':i,'side_m':.06,'yaw_deg':0,'center_m':{'x':x,'y':y}} for i,(x,y) in enumerate([(-.1,-.1),(-.1,.1),(.1,-.1),(.1,.1)],1)]}
        detector=PhysicalPadDetector(manifest)
        K=np.array([[684.,0,668.],[0,686.,353.],[0,0,1.]])
        D=np.array([-.348,.139,.00094,-.00045,-.0276])
        image=np.zeros((720,1280),np.uint8);image[:,1100:]=255
        cropped, Kc=center_crop(image,K)
        self.assertEqual(cropped.shape,(720,720))
        self.assertEqual(int(cropped.max()),0)
        self.assertEqual(Kc[0,2],388.)
        self.assertEqual(K[0,2],668.)
        rv=np.array([2.6,.2,-.1]);tv=np.array([.1,.03,1.2])
        corners=[cv2.projectPoints(detector.models[i],rv,tv,K,D)[0]-np.array([280.,0.]) for i in range(1,5)]
        result=detector.estimate(corners,list(range(1,5)),Kc,D)
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result['camera_from_pad'][:3,3],tv,atol=1e-5)
        single=detector.estimate(corners[:1],[1],Kc,D)
        self.assertIsNotNone(single)
        np.testing.assert_allclose(single['camera_from_pad'][:3,3],tv,atol=1e-5)
        np.testing.assert_allclose(result['camera_from_pad'][:3,:3],cv2.Rodrigues(rv)[0],atol=1e-5)
        with self.assertRaises(ValueError):center_crop(image,K,1281,720)

    def test_learns_current_pad_then_survives_mocap_loss_and_resets(self):
        a=SessionAlignment(min_samples=5,min_duration_s=.2,window_size=20)
        Y=pose([2,-1,.2],.3)
        for t in np.arange(0,.7,.01):a.add_mocap(t,Y@pose([.1*t,0,1]))
        for t in np.arange(.03,.5,.03):out=a.observe(t,pose([.1*t,0,1]))
        self.assertTrue(a.ready)
        self.assertTrue(np.allclose(a.transform,Y))
        self.assertTrue(np.allclose(a.observe(20,pose([0,0,.5])),Y@pose([0,0,.5])))
        a.reset();self.assertFalse(a.ready);self.assertIsNone(a.observe(20,pose([0,0,1])))
        Z=pose([-3,2,.1],-.8)
        for t in np.arange(21,21.7,.01):a.add_mocap(t,Z@pose([.1,0,1]))
        for t in np.arange(21.03,21.5,.03):a.observe(t,pose([.1,0,1]))
        self.assertTrue(a.ready);self.assertTrue(np.allclose(a.transform,Z))

    def test_rejects_unbracketed_jumps_and_bad_transform(self):
        a=SessionAlignment(min_samples=3,min_duration_s=.1,window_size=10)
        a.add_mocap(10,pose([0,0,1]));a.add_mocap(10.01,pose([1,0,1]))
        self.assertIsNone(a.interpolate(10.005))
        self.assertIsNone(a.interpolate(9));self.assertIsNone(a.interpolate(11))
        bad=np.eye(4);bad[0,0]=2;self.assertFalse(a.add_mocap(11,bad))

    def test_full_board_pnp_and_camera_mount_direction(self):
        manifest={'dictionary':'DICT_7X7_50','markers':[{'id':i,'side_m':.06,'yaw_deg':0,'center_m':{'x':x,'y':y}} for i,(x,y) in enumerate([(-.1,-.1),(-.1,.1),(.1,-.1),(.1,.1)],1)]}
        detector=PhysicalPadDetector(manifest)
        K=np.array([[680.,0,640],[0,680.,360],[0,0,1]]);D=np.zeros(5)
        R=cv2.Rodrigues(np.array([2.6,.2,-.1]))[0]
        expected=np.eye(4);expected[:3,:3]=R;expected[:3,3]=[.1,.03,1.2]
        corners=[cv2.projectPoints(detector.models[i],cv2.Rodrigues(R)[0],expected[:3,3],K,D)[0] for i in range(1,5)]
        result=detector.estimate(corners,list(range(1,5)),K,D)
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result['camera_from_pad'],expected,atol=1e-5)
        X=pose([.02,-.12,.01],.1)
        PB=inverse(expected)@inverse(X)
        np.testing.assert_allclose(X@expected@PB,np.eye(4),atol=1e-9)
        self.assertIsNotNone(detector.estimate(corners[:2],[1,2],K,D))
        single=detector.estimate(corners[:1],[1],K,D)
        self.assertIsNotNone(single)
        np.testing.assert_allclose(single['camera_from_pad'],expected,atol=1e-5)
        self.assertIsNone(detector.estimate([],[],K,D))
        self.assertIsNone(detector.estimate(corners,[1,2,2,4],K,D))

    def test_covariance_propagates_orientation_lever_arm(self):
        T=pose([0,0,2]);P=np.diag([.01**2]*3+[.02**2]*3)
        output=relative_covariance(T,np.eye(4),P)
        self.assertAlmostEqual(output[0,0],.01**2+4*.02**2)
        self.assertAlmostEqual(output[2,2],.01**2)
        self.assertGreaterEqual(np.linalg.eigvalsh(output).min(),0)


if __name__=='__main__':unittest.main()
