"""
Stationary Detector for Zero Velocity Updates (ZUPT).
Detects when a vehicle is at a standstill (traffic signals, stop signs, parking)
using rolling statistics on accelerometer variance, gyroscope norms, and velocity gates.
"""

from collections import deque
import numpy as np


class StationaryDetector:
    """
    Robust vehicle stop detector based on Generalized Likelihood Ratio Test (GLRT),
    acceleration magnitude variance, and kinematic speed gating.
    """

    def __init__(
        self,
        window_size: int = 25,             # ~0.5s at 50Hz
        accel_var_threshold: float = 0.008, # (m/s^2)^2 variance threshold
        gyro_norm_threshold: float = 0.03,  # rad/s max gyro norm
        speed_gate_threshold: float = 1.2,  # m/s (~4.3 km/h) max estimated speed for stop
        consecutive_required: int = 8,     # debounce frames to avoid false triggers
    ):
        self.window_size = window_size
        self.accel_var_thresh = accel_var_threshold
        self.gyro_norm_thresh = gyro_norm_threshold
        self.speed_gate_thresh = speed_gate_threshold
        self.consecutive_required = consecutive_required

        self.accel_buffer = deque(maxlen=window_size)
        self.gyro_buffer = deque(maxlen=window_size)
        self.consecutive_count = 0
        self.is_stationary = False

    def add_sample(
        self,
        accel: np.ndarray,
        gyro: np.ndarray,
        current_speed: float = None,
    ) -> bool:
        """
        Ingests a new IMU sample [accel (3,), gyro (3,)] and optional speed estimate.
        Returns True if the vehicle is confirmed to be stationary.
        """
        self.accel_buffer.append(accel)
        self.gyro_buffer.append(gyro)

        if len(self.accel_buffer) < self.window_size:
            self.is_stationary = False
            return False

        # Speed gate: if currently moving fast, reject stop false-positives immediately
        if current_speed is not None and abs(current_speed) > self.speed_gate_thresh:
            self.consecutive_count = 0
            self.is_stationary = False
            return False

        accels = np.array(self.accel_buffer)  # shape (N, 3)
        gyros = np.array(self.gyro_buffer)    # shape (N, 3)

        # Compute variance of acceleration magnitude
        accel_norms = np.linalg.norm(accels, axis=1)
        accel_var = np.var(accel_norms)

        # Compute mean gyro norm
        gyro_norms = np.linalg.norm(gyros, axis=1)
        mean_gyro_norm = np.mean(gyro_norms)

        # Condition for stationary
        stationary_condition = (accel_var < self.accel_var_thresh) and (mean_gyro_norm < self.gyro_norm_thresh)

        if stationary_condition:
            self.consecutive_count += 1
            if self.consecutive_count >= self.consecutive_required:
                self.is_stationary = True
        else:
            self.consecutive_count = 0
            self.is_stationary = False

        return self.is_stationary

    def reset(self):
        """Clear buffers and state."""
        self.accel_buffer.clear()
        self.gyro_buffer.clear()
        self.consecutive_count = 0
        self.is_stationary = False
