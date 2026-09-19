"""Metric planar pad poses and session-only online global alignment (no ROS)."""
from collections import deque
import math
import cv2
import numpy as np
from aruco_landing.pose_alignment import (
    PadAlignmentEstimator, pose_matrix, matrix_pose, normalize_quaternion,
)

CORNERS = np.array([[-.5, .5], [.5, .5], [.5, -.5], [-.5, -.5]])


def inverse(T):
    result = np.eye(4)
    result[:3, :3] = T[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ T[:3, 3]
    return result


def valid_transform(T):
    T = np.asarray(T)
    return (T.shape == (4, 4) and np.isfinite(T).all()
            and np.allclose(T[3], [0, 0, 0, 1])
            and np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-5)
            and np.linalg.det(T[:3, :3]) > .999)


def slerp(a, b, fraction):
    a, b = normalize_quaternion(a), normalize_quaternion(b)
    dot = float(a @ b)
    if dot < 0:
        b, dot = -b, -dot
    if dot > .9995:
        return normalize_quaternion(a + fraction * (b - a))
    angle = math.acos(np.clip(dot, -1, 1))
    return (math.sin((1-fraction)*angle)*a + math.sin(fraction*angle)*b) / math.sin(angle)


def relative_covariance(camera_from_pad, camera_from_body, covariance):
    """Propagate additive translation / fixed-axis orientation errors, including lever arm."""
    R = camera_from_pad[:3, :3].T
    v = camera_from_body[:3, 3] - camera_from_pad[:3, 3]
    skew = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    J = np.zeros((6, 6))
    J[:3, :3] = -R
    J[:3, 3:] = R @ skew
    J[3:, 3:] = -R
    return J @ covariance @ J.T


class PhysicalPadDetector:
    def __init__(self, manifest, min_markers=3, max_rms_px=2.5):
        dictionary_id = getattr(cv2.aruco, manifest['dictionary'])
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self.params = (cv2.aruco.DetectorParameters() if hasattr(cv2.aruco, 'ArucoDetector')
                       else cv2.aruco.DetectorParameters_create())
        self.params.adaptiveThreshWinSizeMax = 101
        self.params.adaptiveThreshWinSizeStep = 4
        self.params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.params.cornerRefinementWinSize = 3
        self.params.cornerRefinementMaxIterations = 50
        self.params.cornerRefinementMinAccuracy = .01
        self.detector = (cv2.aruco.ArucoDetector(self.dictionary, self.params)
                         if hasattr(cv2.aruco, 'ArucoDetector') else None)
        self.min_markers, self.max_rms_px = min_markers, max_rms_px
        self.models = {}
        for m in manifest['markers']:
            angle = np.radians(m['yaw_deg'])
            c, s = np.cos(angle), np.sin(angle)
            points = CORNERS @ np.array([[c, s], [-s, c]]) * m['side_m']
            points += [m['center_m']['x'], m['center_m']['y']]
            if m['id'] in self.models or m['side_m'] <= 0:
                raise ValueError('duplicate ID or invalid metric marker size')
            self.models[m['id']] = np.c_[points, np.zeros(4)]

    def detect(self, gray, K, D):
        if self.detector:
            corners, ids, _ = self.detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.params)
        detected = [] if ids is None else ids.flatten().tolist()
        result = self.estimate(corners, detected, K, D)
        return result, corners, detected

    def estimate(self, corners, ids, K, D):
        if len(set(ids)) != len(ids):
            return None
        used, objects, pixels = [], [], []
        for corner, mid in zip(corners, ids):
            points = np.asarray(corner).reshape(4, 2)
            if mid not in self.models or np.linalg.norm(points-np.roll(points,-1,axis=0),axis=1).mean() < 12:
                continue
            used.append(mid); objects.extend(self.models[mid]); pixels.extend(points)
        if len(used) < self.min_markers:
            return None
        obj, pix = np.array(objects, np.float64), np.array(pixels, np.float64)
        solutions = cv2.solvePnPGeneric(obj, pix, K, D, flags=cv2.SOLVEPNP_IPPE, reprojectionError=np.zeros((2,1),np.float64))
        choices = []
        for rv, tv in zip(solutions[1], solutions[2]):
            rv, tv = cv2.solvePnPRefineLM(obj, pix, K, D, rv, tv)
            error = np.linalg.norm(cv2.projectPoints(obj, rv, tv, K, D)[0].reshape(-1,2)-pix, axis=1)
            if np.isfinite(error).all() and tv[2, 0] > 0:
                choices.append((np.median(error)+np.mean(np.minimum(error,10)), rv, tv, error))
        if not choices:
            return None
        _, rv, tv, error = min(choices, key=lambda x:x[0])
        good = np.flatnonzero(error < 4.)
        if len(good) < .75 * len(obj):
            return None
        rv, tv = cv2.solvePnPRefineLM(obj[good], pix[good], K, D, rv, tv)
        error = np.linalg.norm(cv2.projectPoints(obj, rv, tv, K, D)[0].reshape(-1,2)-pix, axis=1)
        rms = float(np.sqrt(np.mean(error[good]**2)))
        inlier_ids = [mid for i, mid in enumerate(used) if np.all(error[4*i:4*i+4] < 4.)]
        if rms > self.max_rms_px or len(inlier_ids) < self.min_markers:
            return None
        T = np.eye(4); T[:3,:3] = cv2.Rodrigues(rv)[0]; T[:3,3] = tv.ravel()
        # Pad +Z points out of the visible face; require camera above that plane.
        if not valid_transform(T) or inverse(T)[2,3] <= 0:
            return None
        covariance = np.diag([.01**2]*3 + [np.radians(2.)**2]*3)
        return {'camera_from_pad':T, 'inlier_ids':inlier_ids, 'rms_px':rms, 'covariance':covariance}


class SessionAlignment:
    """Learn once per session, then freeze without any saved global-pad transform."""
    def __init__(self, min_samples=30, min_duration_s=2., max_gap_s=.04,
                 max_pair_step_m=.08, max_pair_step_deg=15., **kwargs):
        self.estimator = PadAlignmentEstimator(min_samples=min_samples, **kwargs)
        self.min_duration_s, self.max_gap_s = min_duration_s, max_gap_s
        self.max_pair_step_m, self.max_pair_step_deg = max_pair_step_m, max_pair_step_deg
        self.mocap = deque(maxlen=1000)
        self.first_pair = None
        self.last_pair = None
        self.ready = False
        self.pair_count = 0
        self.reference_time = None
        self.transform = None

    def reset(self):
        self.estimator.clear()
        self.first_pair = self.last_pair = self.transform = self.reference_time = None
        self.ready = False
        self.pair_count = 0
        self.mocap.clear()

    def add_mocap(self, stamp, transform):
        if not np.isfinite(stamp) or not valid_transform(transform):
            return False
        if self.mocap and stamp <= self.mocap[-1][0]:
            return False
        self.mocap.append((stamp, transform.copy()))
        return True

    def interpolate(self, stamp):
        if len(self.mocap) < 2 or stamp < self.mocap[0][0] or stamp > self.mocap[-1][0]:
            return None
        for i in range(len(self.mocap)-1, 0, -1):
            t1, B = self.mocap[i]
            t0, A = self.mocap[i-1]
            if t0 <= stamp <= t1:
                if t1-t0 > self.max_gap_s:
                    return None
                p0, q0 = matrix_pose(A); p1, q1 = matrix_pose(B)
                angle = math.degrees(2*math.acos(np.clip(abs(float(q0@q1)),0,1)))
                if np.linalg.norm(p1-p0) > self.max_pair_step_m or angle > self.max_pair_step_deg:
                    return None
                f = (stamp-t0)/(t1-t0)
                return pose_matrix((1-f)*p0+f*p1, slerp(q0,q1,f))
        return None

    def observe(self, stamp, pad_from_body):
        if not valid_transform(pad_from_body):
            return None
        if not self.ready:
            global_body = self.interpolate(stamp)
            if global_body is None or (self.last_pair is not None and stamp <= self.last_pair):
                return None
            # Do not combine fragments separated by long observation gaps.
            if self.last_pair is not None and stamp-self.last_pair > 1.:
                self.estimator.clear(); self.first_pair = None; self.pair_count = 0
            if self.first_pair is None:
                self.first_pair = stamp
            self.last_pair = stamp; self.pair_count += 1
            estimate = self.estimator.add(global_body, pad_from_body)
            if estimate['ready'] and stamp-self.first_pair >= self.min_duration_s:
                self.transform = self.estimator.freeze()['transform'].copy()
                self.reference_time = stamp
                self.ready = True
        return self.transform @ pad_from_body if self.ready else None
