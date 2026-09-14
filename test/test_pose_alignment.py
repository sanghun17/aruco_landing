#!/usr/bin/env python3

import math
import unittest

import numpy as np

from aruco_landing.pose_alignment import (
    PadAlignmentEstimator,
    aligned_global_body,
    matrix_pose,
    pose_matrix,
    quaternion_distance_deg,
)


def yaw_quaternion(angle):
    return (0.0, 0.0, math.sin(angle / 2.0), math.cos(angle / 2.0))


class PoseAlignmentTest(unittest.TestCase):
    def test_recovers_global_pad_transform_and_preserves_pose_at_transition(self):
        expected = pose_matrix((2.0, -1.5, 0.2), yaw_quaternion(0.35))
        estimator = PadAlignmentEstimator(
            min_samples=20,
            translation_outlier_m=0.04,
            rotation_outlier_deg=4.0,
            max_translation_std_m=0.01,
            max_rotation_std_deg=1.0,
        )
        generator = np.random.default_rng(7)
        last_global_body = None
        last_pad_body = None
        for index in range(60):
            pad_body = pose_matrix(
                (0.6 - 0.008 * index, -0.25 + 0.004 * index, 1.4 - 0.01 * index),
                yaw_quaternion(-0.2 + 0.003 * index),
            )
            global_body = np.matmul(expected, pad_body)
            global_body[0:3, 3] += generator.normal(0.0, 0.001, 3)
            estimator.add(global_body, pad_body)
            last_global_body = global_body
            last_pad_body = pad_body

        # A gross marker outlier must not perturb the estimate.
        estimator.add(
            last_global_body,
            pose_matrix((8.0, -4.0, 2.0), yaw_quaternion(2.5)),
        )
        estimate = estimator.freeze()
        self.assertTrue(estimate["ready"])
        self.assertTrue(estimate["frozen"])
        self.assertGreaterEqual(estimate["inlier_count"], 59)

        translation, quaternion = matrix_pose(estimate["transform"])
        expected_translation, expected_quaternion = matrix_pose(expected)
        self.assertLess(np.linalg.norm(translation - expected_translation), 0.002)
        self.assertLess(
            quaternion_distance_deg(quaternion, expected_quaternion), 0.1
        )

        transitioned = aligned_global_body(estimate["transform"], last_pad_body)
        self.assertLess(
            np.linalg.norm(transitioned[0:3, 3] - last_global_body[0:3, 3]),
            0.004,
        )

    def test_refuses_freeze_before_minimum_sample_count(self):
        estimator = PadAlignmentEstimator(min_samples=3, window_size=3)
        identity = np.eye(4)
        estimator.add(identity, identity)
        estimator.add(identity, identity)
        with self.assertRaises(ValueError):
            estimator.freeze()


if __name__ == "__main__":
    unittest.main()
