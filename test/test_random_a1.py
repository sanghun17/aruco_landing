"""Metric geometry and pose checks for the printed A1 random pad."""
from pathlib import Path
import cv2
import numpy as np
import yaml
from aruco_landing.campaign import pad_identity
from aruco_landing.physical_pad import PhysicalPadDetector


def test_random_a1_print_geometry_and_original_pnp():
    path=Path(__file__).resolve().parents[1]/'config/random_a1_700mm_runtime.yaml'
    manifest=yaml.safe_load(path.read_text())
    detector=PhysicalPadDetector(manifest)
    assert manifest['dictionary']=='DICT_4X4_100'
    assert pad_identity(path)['key']=='random_a1'
    assert set(detector.models)=={0,1}
    for mid,x,y,side in [(0,445.22226150027524,145.5681359288536,213.5),
                         (1,597.3669747873828,387.9759390192812,57.4)]:
        # Printed top-left corner maps to paper-up, paper-left coordinates.
        np.testing.assert_allclose(detector.models[mid][0],[(350-y)/1000,(350-x)/1000,0],atol=1e-12)
        assert abs(np.linalg.norm(detector.models[mid][1]-detector.models[mid][0])-side/1000)<1e-12
    K=np.array([[684.,0,360.],[0,686.,360.],[0,0,1.]])
    D=np.array([-.348,.139,.00094,-.00045,-.0276])
    rv=np.array([2.8,.08,-.1])
    for ids,height in [([0,1],1.5),([0],1.5),([1],.5)]:
        tv=np.array([.02,-.01,height])
        corners=[cv2.projectPoints(detector.models[mid],rv,tv,K,D)[0] for mid in ids]
        result=detector.estimate(corners,ids,K,D)
        assert result is not None
        np.testing.assert_allclose(result['camera_from_pad'][:3,3],tv,atol=1e-5)
        np.testing.assert_allclose(result['camera_from_pad'][:3,:3],cv2.Rodrigues(rv)[0],atol=1e-5)
