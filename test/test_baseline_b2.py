"""Metric pose and board-frame checks for the actual 700mm B2 print."""
from pathlib import Path
import cv2
import numpy as np
import yaml
from aruco_landing.physical_pad import PhysicalPadDetector


def test_print_geometry_and_center_marker_metric_pose():
    path=Path(__file__).resolve().parents[1]/'config/baseline_b2_700mm_runtime.yaml'
    manifest=yaml.safe_load(path.read_text())
    detector=PhysicalPadDetector(manifest)
    assert manifest['dictionary']=='DICT_6X6_50'
    assert set(detector.models)=={1,2,3,4,19,43}
    # Source figure has ID43 centered, 23 units on a 341-unit, 700mm canvas.
    np.testing.assert_allclose(detector.models[43].mean(axis=0),[0,0,0],atol=1e-12)
    assert abs(np.linalg.norm(detector.models[43][1]-detector.models[43][0])-.7*23/341)<1e-12
    # Upper-left of the printed board is +X,+Y in the pad frame.
    assert np.all(detector.models[1].mean(axis=0)[:2]>0)
    assert np.all(detector.models[3].mean(axis=0)[:2]<0)
    K=np.array([[684.,0,360.],[0,686.,360.],[0,0,1.]])
    D=np.array([-.348,.139,.00094,-.00045,-.0276])
    rv=np.array([2.8,.08,-.1])
    for ids,height in [([43],.3),([1,2,3,4,43],1.5)]:
        tv=np.array([.02,-.01,height])
        corners=[cv2.projectPoints(detector.models[i],rv,tv,K,D)[0] for i in ids]
        pose=detector.estimate(corners,ids,K,D)
        assert pose is not None
        np.testing.assert_allclose(pose['camera_from_pad'][:3,3],tv,atol=1e-5)
        np.testing.assert_allclose(pose['camera_from_pad'][:3,:3],cv2.Rodrigues(rv)[0],atol=1e-5)
