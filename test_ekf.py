"""
Unit tests for Kinematics, ESKF, and Stationary Detector.
"""

import unittest
import numpy as np

from src.kinematics.quaternion import (
    quat_normalize,
    quat_multiply,
    quat_to_rot_matrix,
    rot_vector_to_quat,
    euler_from_quat,
    quat_from_euler,
)
from src.filters.eskf import ErrorStateKalmanFilter
from src.filters.stationary_detector import StationaryDetector


class TestKinematics(unittest.TestCase):
    def test_quaternion_identity(self):
        q = np.array([1.0, 0.0, 0.0, 0.0])
        R = quat_to_rot_matrix(q)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-7)

    def test_euler_roundtrip(self):
        roll, pitch, yaw = 0.1, -0.2, 0.8
        q = quat_from_euler(roll, pitch, yaw)
        r_rec, p_rec, y_rec = euler_from_quat(q)
        self.assertAlmostEqual(roll, r_rec, places=5)
        self.assertAlmostEqual(pitch, p_rec, places=5)
        self.assertAlmostEqual(yaw, y_rec, places=5)

    def test_vector_rotation_90deg(self):
        # Yaw 90 degrees: [1, 0, 0] should rotate to [0, 1, 0]
        q = quat_from_euler(0.0, 0.0, np.pi / 2.0)
        R = quat_to_rot_matrix(q)
        v_world = R @ np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(v_world, [0.0, 1.0, 0.0], atol=1e-7)


class TestESKF(unittest.TestCase):
    def test_stationary_prediction_gravity_cancellation(self):
        eskf = ErrorStateKalmanFilter()
        # Level vehicle stationary: specific force is [0, 0, 9.80665]
        accel = np.array([0.0, 0.0, 9.80665])
        gyro = np.array([0.0, 0.0, 0.0])

        for _ in range(50):
            eskf.predict(accel, gyro, dt=0.02)

        # Level car at rest should not move
        np.testing.assert_allclose(eskf.v, [0.0, 0.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(eskf.p, [0.0, 0.0, 0.0], atol=1e-5)

    def test_nhc_constrains_lateral_velocity(self):
        eskf = ErrorStateKalmanFilter()
        # Artificially inject sideways velocity
        eskf.v = np.array([10.0, 5.0, 2.0])
        self.assertGreater(abs(eskf.body_velocity[1]), 1.0)

        # Apply NHC multiple times
        for _ in range(5):
            eskf.update_nhc()

        # Lateral and vertical body velocity should be driven toward zero
        v_b = eskf.body_velocity
        self.assertAlmostEqual(v_b[1], 0.0, delta=0.2)
        self.assertAlmostEqual(v_b[2], 0.0, delta=0.2)

    def test_zupt_zeros_velocity(self):
        eskf = ErrorStateKalmanFilter()
        eskf.v = np.array([3.0, -1.0, 0.5])
        for _ in range(3):
            eskf.update_zupt()
        np.testing.assert_allclose(eskf.v, [0.0, 0.0, 0.0], atol=0.1)


class TestStationaryDetector(unittest.TestCase):
    def test_detects_stop_and_moving(self):
        detector = StationaryDetector(window_size=20, accel_var_threshold=0.01, consecutive_required=5)
        rng = np.random.default_rng(42)

        # Stationary samples: low variance
        for _ in range(30):
            accel = np.array([0.0, 0.0, 9.81]) + rng.normal(0.0, 0.01, 3)
            gyro = rng.normal(0.0, 0.002, 3)
            is_stat = detector.add_sample(accel, gyro, current_speed=0.0)

        self.assertTrue(is_stat)

        # Moving fast: speed gate blocks stationary trigger immediately
        for _ in range(10):
            accel = np.array([0.0, 0.0, 9.81]) + rng.normal(0.0, 0.01, 3)
            gyro = rng.normal(0.0, 0.002, 3)
            is_stat = detector.add_sample(accel, gyro, current_speed=12.0)

        self.assertFalse(is_stat)


if __name__ == "__main__":
    unittest.main()
