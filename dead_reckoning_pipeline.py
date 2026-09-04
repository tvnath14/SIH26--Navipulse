"""
Dead Reckoning Pipeline Orchestrator.
Fuses IMU streams, Stationary Detection (ZUPT), Non-Holonomic Constraints (NHC),
AI Speed Inferences, and GNSS Updates into a unified navigation state.
"""

import time
from collections import deque
from enum import Enum
import numpy as np

from src.filters.eskf import ErrorStateKalmanFilter
from src.filters.stationary_detector import StationaryDetector
from src.kinematics.quaternion import quat_from_euler


class NavigationMode(Enum):
    GNSS_FIX = "GNSS_FIX"
    DEAD_RECKONING = "DEAD_RECKONING"
    STATIONARY_LOCK = "STATIONARY_LOCK"


class SlidingWindowBuffer:
    """Buffer keeping recent IMU frames for AI feature extraction."""
    def __init__(self, window_size: int = 50):
        self.buffer = deque(maxlen=window_size)

    def add(self, frame: np.ndarray):
        self.buffer.append(frame)

    def is_ready(self) -> bool:
        return len(self.buffer) == self.buffer.maxlen

    def as_array(self) -> np.ndarray:
        return np.array(self.buffer)


class DeadReckoningPipeline:
    """
    Real-time navigation engine coordinating ESKF, Kinematic Constraints, and AI inferences.
    """

    def __init__(
        self,
        sample_hz: float = 50.0,
        window_size: int = 50,
        enable_nhc: bool = True,
        enable_zupt: bool = True,
        ai_speed_estimator = None,
        init_pos: np.ndarray = None,
        init_yaw: float = 0.0,
    ):
        self.dt = 1.0 / sample_hz
        self.enable_nhc = enable_nhc
        self.enable_zupt = enable_zupt
        self.ai_speed_estimator = ai_speed_estimator

        self.buffer = SlidingWindowBuffer(window_size)
        self.detector = StationaryDetector(window_size=int(sample_hz * 0.5))

        init_quat = quat_from_euler(0.0, 0.0, init_yaw) if init_yaw != 0.0 else None
        self.eskf = ErrorStateKalmanFilter(init_pos=init_pos, init_quat=init_quat)

        self.current_mode = NavigationMode.DEAD_RECKONING
        self.total_distance = 0.0
        self.outage_distance_m = 0.0
        self.last_position = self.eskf.p.copy()
        self.outage_duration_s = 0.0

    def step(
        self,
        accel: np.ndarray,
        gyro: np.ndarray,
        gnss_pos: np.ndarray = None,
        gnss_vel: np.ndarray = None,
    ) -> dict:
        """
        Processes a single IMU frame [ax, ay, az, gx, gy, gz].
        Executes in < 5ms (well under the 20ms threshold for 50Hz).
        """
        t_start = time.perf_counter()

        frame = np.concatenate([accel, gyro])
        self.buffer.add(frame)

        # 1. Stationary detection (ZUPT condition)
        est_forward_speed = float(self.eskf.body_velocity[0])
        is_stationary = self.detector.add_sample(accel, gyro, current_speed=est_forward_speed) if self.enable_zupt else False

        # 2. ESKF Propagation (Strapdown inertial mechanics + gravity removal)
        self.eskf.predict(accel, gyro, self.dt)

        # 3. Measurement corrections
        if gnss_pos is not None:
            # Open sky GNSS aided navigation
            self.eskf.update_gnss(gnss_pos, gnss_vel)
            self.current_mode = NavigationMode.GNSS_FIX
            self.outage_duration_s = 0.0
        elif is_stationary and self.enable_zupt:
            # Vehicle stopped at signal/traffic -> zero out drift and lock gyro bias
            self.eskf.update_zupt(gyro_raw=gyro)
            self.current_mode = NavigationMode.STATIONARY_LOCK
            self.outage_duration_s += self.dt
        else:
            # Pure Dead Reckoning during GNSS blackout
            self.current_mode = NavigationMode.DEAD_RECKONING
            self.outage_duration_s += self.dt

            # Apply Non-Holonomic Constraints (lateral vy=0, vertical vz=0)
            if self.enable_nhc:
                self.eskf.update_nhc()

            # Apply AI Forward Speed update if model and window are ready
            if self.ai_speed_estimator is not None and self.buffer.is_ready():
                ai_speed = self.ai_speed_estimator(self.buffer.as_array())
                if ai_speed is not None and ai_speed >= 0.0:
                    self.eskf.update_ai_speed(ai_speed)

        # Update distance traveled
        delta_p = np.linalg.norm(self.eskf.p - self.last_position)
        self.total_distance += delta_p
        if gnss_pos is not None:
            self.outage_distance_m = 0.0
        else:
            self.outage_distance_m += delta_p
        self.last_position = self.eskf.p.copy()

        t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        roll_deg, pitch_deg, yaw_deg = self.eskf.euler_angles

        return {
            "position": self.eskf.p.copy(),
            "velocity": self.eskf.v.copy(),
            "body_velocity": self.eskf.body_velocity.copy(),
            "heading_deg": yaw_deg,
            "roll_deg": roll_deg,
            "pitch_deg": pitch_deg,
            "mode": self.current_mode.value,
            "is_stationary": is_stationary,
            "total_distance_m": self.total_distance,
            "outage_distance_m": self.outage_distance_m,
            "outage_duration_s": self.outage_duration_s,
            "latency_ms": t_elapsed_ms,
        }
