"""Rigid registration between a global pose source and a landing-pad frame."""

from collections import deque
import math

import numpy as np

from aruco_landing.geometry import rotation_matrix_to_quaternion


def normalize_quaternion(quaternion):
    value = np.asarray(quaternion, dtype=float)
    if value.shape != (4,):
        raise ValueError("quaternion must have four elements")
    norm = float(np.linalg.norm(value))
    if norm < 1e-12:
        raise ValueError("quaternion norm is zero")
    return value / norm


def pose_matrix(position, quaternion):
    """Return parent-from-child homogeneous transform for an xyzw pose."""
    x, y, z, w = normalize_quaternion(quaternion)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    transform = np.eye(4, dtype=float)
    transform[0:3, 0:3] = np.asarray([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ])
    transform[0:3, 3] = np.asarray(position, dtype=float)
    return transform


def matrix_pose(transform):
    matrix = np.asarray(transform, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError("transform must be 4x4")
    return (
        matrix[0:3, 3].copy(),
        np.asarray(rotation_matrix_to_quaternion(matrix[0:3, 0:3])),
    )


def average_quaternions(quaternions):
    values = [normalize_quaternion(value) for value in quaternions]
    if not values:
        raise ValueError("at least one quaternion is required")
    accumulator = sum(np.outer(value, value) for value in values)
    _, eigenvectors = np.linalg.eigh(accumulator)
    result = eigenvectors[:, -1]
    if result[3] < 0.0:
        result = -result
    return normalize_quaternion(result)


def quaternion_distance_deg(first, second):
    dot = abs(float(np.dot(
        normalize_quaternion(first), normalize_quaternion(second)
    )))
    return math.degrees(2.0 * math.acos(min(1.0, max(-1.0, dot))))


class PadAlignmentEstimator:
    """Robustly estimate global-from-pad from synchronized body poses.

    Each observation contains the same rigid body expressed once in the global
    frame and once in the pad frame. Therefore::

        T_global_pad = T_global_body * inverse(T_pad_body)
    """

    def __init__(
            self, min_samples=30, window_size=240,
            translation_outlier_m=0.08, rotation_outlier_deg=10.0,
            max_translation_std_m=0.03, max_rotation_std_deg=5.0):
        if min_samples < 1 or window_size < min_samples:
            raise ValueError("window_size must be >= min_samples >= 1")
        self.min_samples = int(min_samples)
        self.translation_outlier_m = float(translation_outlier_m)
        self.rotation_outlier_deg = float(rotation_outlier_deg)
        self.max_translation_std_m = float(max_translation_std_m)
        self.max_rotation_std_deg = float(max_rotation_std_deg)
        self._candidates = deque(maxlen=int(window_size))
        self._frozen = False
        self._frozen_estimate = None

    @property
    def frozen(self):
        return self._frozen

    def clear(self):
        self._candidates.clear()
        self._frozen = False
        self._frozen_estimate = None

    def add(self, global_from_body, pad_from_body):
        if self._frozen:
            return self.estimate()
        candidate = np.matmul(
            np.asarray(global_from_body, dtype=float),
            np.linalg.inv(np.asarray(pad_from_body, dtype=float)),
        )
        if not np.all(np.isfinite(candidate)):
            raise ValueError("alignment candidate contains non-finite values")
        self._candidates.append(candidate)
        return self.estimate()

    def freeze(self):
        estimate = self.estimate()
        if estimate is None or not estimate["ready"]:
            raise ValueError("alignment is not ready")
        self._frozen_estimate = dict(estimate, frozen=True)
        self._frozen = True
        return self._frozen_estimate

    def set_frozen_transform(self, transform, metadata=None):
        """Install a previously surveyed or bag-derived transform."""
        value = np.asarray(transform, dtype=float)
        if value.shape != (4, 4) or not np.all(np.isfinite(value)):
            raise ValueError("loaded alignment transform must be finite and 4x4")
        estimate = {
            "transform": value.copy(),
            "sample_count": 0,
            "inlier_count": 0,
            "translation_std_m": None,
            "translation_p95_m": None,
            "rotation_std_deg": None,
            "rotation_p95_deg": None,
            "ready": True,
            "frozen": True,
            "loaded": True,
        }
        if metadata:
            estimate["loaded_metadata"] = dict(metadata)
        self._frozen_estimate = estimate
        self._frozen = True
        return estimate

    def estimate(self):
        if self._frozen and self._frozen_estimate is not None:
            return self._frozen_estimate
        if not self._candidates:
            return None

        translations = np.asarray([
            candidate[0:3, 3] for candidate in self._candidates
        ])
        quaternions = [matrix_pose(candidate)[1] for candidate in self._candidates]
        initial_translation = np.median(translations, axis=0)
        initial_quaternion = average_quaternions(quaternions)
        translation_residuals = np.linalg.norm(
            translations - initial_translation, axis=1
        )
        rotation_residuals = np.asarray([
            quaternion_distance_deg(value, initial_quaternion)
            for value in quaternions
        ])
        inlier_mask = np.logical_and(
            translation_residuals <= self.translation_outlier_m,
            rotation_residuals <= self.rotation_outlier_deg,
        )
        inlier_indices = np.flatnonzero(inlier_mask)
        if inlier_indices.size:
            translation = np.mean(translations[inlier_indices], axis=0)
            quaternion = average_quaternions([
                quaternions[index] for index in inlier_indices
            ])
        else:
            translation = initial_translation
            quaternion = initial_quaternion

        translation_errors = np.linalg.norm(
            translations[inlier_indices] - translation, axis=1
        ) if inlier_indices.size else np.asarray([], dtype=float)
        rotation_errors = np.asarray([
            quaternion_distance_deg(quaternions[index], quaternion)
            for index in inlier_indices
        ])
        transform = pose_matrix(translation, quaternion)
        translation_std_m = (
            float(np.std(translation_errors)) if translation_errors.size else None
        )
        rotation_std_deg = (
            float(np.std(rotation_errors)) if rotation_errors.size else None
        )
        ready = (
            inlier_indices.size >= self.min_samples
            and translation_std_m <= self.max_translation_std_m
            and rotation_std_deg <= self.max_rotation_std_deg
        )
        return {
            "transform": transform,
            "sample_count": len(self._candidates),
            "inlier_count": int(inlier_indices.size),
            "translation_std_m": translation_std_m,
            "translation_p95_m": (
                float(np.percentile(translation_errors, 95.0))
                if translation_errors.size else None
            ),
            "rotation_std_deg": rotation_std_deg,
            "rotation_p95_deg": (
                float(np.percentile(rotation_errors, 95.0))
                if rotation_errors.size else None
            ),
            "ready": bool(ready),
            "frozen": bool(self._frozen),
        }


def aligned_global_body(global_from_pad, pad_from_body):
    return np.matmul(
        np.asarray(global_from_pad, dtype=float),
        np.asarray(pad_from_body, dtype=float),
    )
