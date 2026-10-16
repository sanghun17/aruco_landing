"""Real bad-quad regression plus exact planar and single-marker contracts."""
import json
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
import yaml
from aruco_landing.physical_pad import PhysicalPadDetector

ROOT = Path(__file__).resolve().parents[1]


def detector():
    return PhysicalPadDetector(yaml.safe_load((ROOT/'config/baseline_b2_700mm_runtime.yaml').read_text()))


def test_bad_raw_quads_recover_or_withhold_without_partial_bad_marker():
    data = json.loads((ROOT/'test/fixtures/b2_bad_quads_20261016.json').read_text())
    d = detector(); K = np.array(data['K']); D = np.array(data['D'])
    recovered = 0
    for frame in data['frames']:
        ids = frame['ids']
        corners = [np.array(frame['corners'][str(mid)], np.float32).reshape(1, 4, 2) for mid in ids]
        result = d.estimate(corners, ids, K, D)
        if result is None:
            continue
        recovered += 1
        assert result['consensus']['per_marker_points']['19'] == 0
        # Frozen recorded registration and mocap are evaluation-only evidence.
        position = np.linalg.inv(result['camera_from_pad'])[:3, 3]
        assert np.linalg.norm(position-frame['gt_camera_position_pad_m']) < .1
    assert recovered >= 13


def test_exact_planar_boards_and_single_marker():
    d = detector(); K = np.array([[684.,0,360.],[0,686.,360.],[0,0,1.]])
    D = np.array([-.348,.139,.00094,-.00045,-.0276])
    for ids, height in [([43], .3), ([1, 2], 1.5), ([1,2,3,4], 1.5), ([1,2,3,4,19,43], 2.)]:
        rv = np.array([2.8,.08,-.1]); tv = np.array([.02,-.01,height])
        corners = [cv2.projectPoints(d.models[mid], rv, tv, K, D)[0] for mid in ids]
        result = d.estimate(corners, ids, K, D)
        assert result is not None
        np.testing.assert_allclose(result['camera_from_pad'][:3,3], tv, atol=1e-5)
        assert result['consensus']['used_points'] == 4*len(ids)
        assert result['consensus']['method'] == ('ippe_lm' if len(ids) == 1 else 'corner_ransac_ippe_lm')


def test_failed_consensus_does_not_fall_back_to_unfiltered_pose():
    d = detector(); K = np.array([[684.,0,360.],[0,686.,360.],[0,0,1.]])
    corners = [cv2.projectPoints(d.models[mid], np.array([2.8,.08,-.1]), np.array([.02,-.01,1.5]), K, np.zeros(5))[0] for mid in [1,2]]
    with patch.object(cv2, 'solvePnPRansac', return_value=(False, None, None, None)):
        assert d.estimate(corners, [1,2], K, np.zeros(5)) is None
    assert not d.last_pnp_diagnostics['accepted']
