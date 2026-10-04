import unittest
import cv2
import numpy as np
import torch
from unittest import mock
from aruco_landing.batched_detection import BatchedPadDetector
from aruco_landing.physical_pad import PhysicalPadDetector


class BatchedDetectionTest(unittest.TestCase):
    def test_grayscale_batch_preserves_reference_pose_and_accounts_transfer(self):
        manifest=dict(dictionary='DICT_4X4_100',markers=[dict(id=7,side_m=.12,yaw_deg=0.,center_m=dict(x=0.,y=0.))])
        K=np.array([[700.,0.,360.],[0.,700.,360.],[0.,0.,1.]])
        image=np.full((720,720),255,np.uint8)
        dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        image[276:444,276:444]=cv2.aruco.generateImageMarker(dictionary,7,168)
        expected=PhysicalPadDetector(manifest).detect(image,K,np.zeros(5))[0]
        rgb=torch.from_numpy(np.repeat(image[None,...,None],3,axis=-1))
        batch=BatchedPadDetector(manifest,1,K)
        actual,statistics=batch.detect(rgb)
        self.assertIsNotNone(expected)
        self.assertIsNotNone(actual[0])
        np.testing.assert_allclose(actual[0]['camera_from_pad'],expected['camera_from_pad'],atol=1e-10)
        self.assertEqual(statistics['transferred_bytes'],720*720)
        self.assertEqual(statistics['markers'],1)
        self.assertEqual(batch.last_detected_ids, [[7]])
        # Detection availability remains measurable when PnP rejects a frame.
        with mock.patch.object(batch.detectors[0], 'estimate', return_value=None):
            rejected,_=batch.detect(rgb)
        self.assertEqual(rejected, [None])
        self.assertEqual(batch.last_detected_ids, [[7]])
        batch.detect(torch.full_like(rgb,255))
        self.assertEqual(batch.last_detected_ids, [[]])


if __name__=='__main__': unittest.main()
