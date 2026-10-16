"""Gravity-aided planar pose geometry, without ROS or a global position input."""
from collections import deque
import math
import cv2
import numpy as np


def _unit(value):
    value = np.asarray(value, dtype=float)
    norm = np.linalg.norm(value)
    if value.shape != (3,) or not np.isfinite(value).all() or norm < 1e-8:
        raise ValueError('invalid gravity direction')
    return value / norm


def rotation_between(first, second):
    a, b = _unit(first), _unit(second)
    cross = np.cross(a, b)
    angle = math.atan2(np.linalg.norm(cross), np.clip(a @ b, -1., 1.))
    if angle < 1e-10:
        return np.eye(3)
    if np.linalg.norm(cross) < 1e-8:
        raise ValueError('opposite gravity directions')
    return cv2.Rodrigues(cross / np.linalg.norm(cross) * angle)[0]


def planar_pose_with_normal(objects, pixels, K, D, camera_normal):
    """Solve planar pose with a measured outward normal; yaw remains visual.

    Rotating undistorted rays to that normal makes the planar projection a 2D
    similarity. Its signed scale preserves the visible face, including the
    usual camera-facing normal near optical -Z. No PnP pitch hypothesis or
    external position is used.
    """
    obj = np.asarray(objects, dtype=float)
    pix = np.asarray(pixels, dtype=float).reshape(-1, 2)
    if len(obj) < 4 or len(obj) != len(pix) or not np.isfinite(pix).all():
        raise ValueError('invalid planar correspondences')
    if not np.allclose(obj[:, 2], 0., atol=1e-8):
        raise ValueError('normal constraint requires a planar metric model')
    normal = _unit(camera_normal)
    seed = np.array([1., 0., 0.]) if abs(normal[0]) < .9 else np.array([0., 1., 0.])
    x = _unit(seed - normal * (seed @ normal))
    basis = np.column_stack((x, np.cross(normal, x), normal))
    rays = np.column_stack((cv2.undistortPoints(pix.reshape(-1, 1, 2), K, D).reshape(-1, 2), np.ones(len(pix)))) @ basis
    if np.any(abs(rays[:, 2]) < .1) or np.any(rays[:, 2] * rays[0, 2] <= 0):
        return None
    rectified = rays[:, :2] / rays[:, 2, None]
    a = obj[:, :2] - obj[:, :2].mean(axis=0)
    b = rectified - rectified.mean(axis=0)
    spread = float(np.sum(a * a))
    if spread < 1e-12:
        return None
    u, _, vt = np.linalg.svd(a.T @ b)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    scale = float(np.sum((a @ rotation.T) * b) / spread)
    if scale < 1e-10:
        return None
    sign = np.sign(np.median(rays[:, 2]))
    rotation *= sign
    depth = sign / scale
    shift = rectified.mean(axis=0) - (obj[:, :2] @ rotation.T).mean(axis=0) / depth
    local_rotation = np.eye(3)
    local_rotation[:2, :2] = rotation
    pose = np.eye(4)
    pose[:3, :3] = basis @ local_rotation
    pose[:3, 3] = basis @ np.r_[shift * depth, depth]
    if not np.isfinite(pose).all() or np.any((obj @ pose[:3, :3].T + pose[:3, 3])[:, 2] <= 0):
        return None
    return pose


def level_alignment_observation(global_body, pad_body):
    """Preserve measured body position while constraining pad/global tilt to zero."""
    candidate_rotation = global_body[:3, :3] @ pad_body[:3, :3].T
    # Closest rotation about world +Z, without second-order roll/pitch yaw.
    yaw = math.atan2(candidate_rotation[1, 0] - candidate_rotation[0, 1],
                     candidate_rotation[0, 0] + candidate_rotation[1, 1])
    c, s = math.cos(yaw), math.sin(yaw)
    level = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    observation = global_body.copy()
    observation[:3, :3] = level @ pad_body[:3, :3]
    return observation


class GravityReference:
    """Learn a small fixed IMU/body up-vector correction, then freeze it.

    Only preflight reference orientation is learned. Subsequent output needs
    IMU attitude alone; reference position and reference yaw are never used.
    This is not an attempt to calibrate unobservable IMU/body yaw.
    """
    def __init__(self, min_samples=60, min_duration_s=2., max_correction_deg=5., max_rms_deg=.5):
        self.min_samples = min_samples
        self.min_duration_s = min_duration_s
        self.max_correction_deg = max_correction_deg
        self.max_rms_deg = max_rms_deg
        self.reset()

    def reset(self):
        self.samples = deque(maxlen=240)
        self.ready = False
        self.rotation = None
        self.correction_deg = None
        self.rms_deg = None

    def observe(self, stamp, global_body_rotation, imu_rotation):
        if self.ready:
            return True
        if not np.isfinite(stamp):
            return False
        if self.samples and stamp <= self.samples[-1][0]:
            return False
        if self.samples and stamp - self.samples[-1][0] > 1.:
            self.reset()
        body_up = _unit(np.asarray(global_body_rotation).T @ [0., 0., 1.])
        imu_up = _unit(np.asarray(imu_rotation).T @ [0., 0., 1.])
        self.samples.append((stamp, body_up, imu_up))
        if len(self.samples) < self.min_samples or stamp - self.samples[0][0] < self.min_duration_s:
            return False
        body = np.array([v[1] for v in self.samples])
        imu = np.array([v[2] for v in self.samples])
        rotation = rotation_between(np.median(imu, axis=0), np.median(body, axis=0))
        errors = np.degrees(np.arccos(np.clip(np.sum((imu @ rotation.T) * body, axis=1), -1., 1.)))
        good = errors < 1.
        if good.sum() < self.min_samples or good.mean() < .8:
            return False
        rotation = rotation_between(np.mean(imu[good], axis=0), np.mean(body[good], axis=0))
        correction = float(np.degrees(np.linalg.norm(cv2.Rodrigues(rotation)[0])))
        errors = np.degrees(np.arccos(np.clip(np.sum((imu[good] @ rotation.T) * body[good], axis=1), -1., 1.)))
        rms = float(np.sqrt(np.mean(errors ** 2)))
        self.correction_deg, self.rms_deg = correction, rms
        if correction > self.max_correction_deg or rms > self.max_rms_deg:
            return False
        self.rotation, self.ready = rotation, True
        return True


class AttitudeHistory:
    """Bracketed interpolation only: no extrapolation across lost IMU samples."""
    def __init__(self, max_gap_s=.04, max_step_deg=15.):
        self.samples = deque(maxlen=1000)
        self.max_gap_s, self.max_step_deg = max_gap_s, max_step_deg

    def clear(self):
        self.samples.clear()

    def add(self, stamp, rotation):
        R = np.asarray(rotation, dtype=float)
        if not np.isfinite(stamp) or R.shape != (3, 3) or not np.isfinite(R).all():
            return False
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-5) or np.linalg.det(R) < .999:
            return False
        if self.samples and stamp <= self.samples[-1][0]:
            return False
        self.samples.append((stamp, R.copy()))
        return True

    def interpolate(self, stamp):
        if len(self.samples) < 2 or not self.samples[0][0] <= stamp <= self.samples[-1][0]:
            return None
        for i in range(len(self.samples)-1, 0, -1):
            t1, R1 = self.samples[i]
            t0, R0 = self.samples[i-1]
            if t0 <= stamp <= t1:
                if t1-t0 > self.max_gap_s:
                    return None
                relative = cv2.Rodrigues(R0.T @ R1)[0]
                if np.degrees(np.linalg.norm(relative)) > self.max_step_deg:
                    return None
                return R0 @ cv2.Rodrigues(relative * ((stamp-t0)/(t1-t0)))[0]
        return None
