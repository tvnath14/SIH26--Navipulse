"""
SIH Dead Reckoning - Backend Prototype (Stage 2: Physics-Based Filter & Comparison)
----------------------------------------------------------------------------------
Stage 1: Naive double integration baseline (unconstrained, rapid drift).
Stage 2: Physics-based Extended Kalman Filter (ESKF) with:
          - Gravity removal & 3D attitude tracking (quaternions)
          - Non-Holonomic Constraints (NHC: vy_body=0, vz_body=0)
          - Zero Velocity Updates (ZUPT: stationary detection at stops)

Run this script to see both pipelines run side-by-side on simulated vehicle motion!
"""

import os
import sys
import time
from collections import deque
import numpy as np

# Add project root to sys.path so src imports resolve cleanly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.filters.eskf import ErrorStateKalmanFilter
from src.filters.stationary_detector import StationaryDetector

# --- CONFIG ---
WINDOW_SIZE = 50          # past sensor frames buffer for AI model
SAMPLE_HZ = 50.0          # simulated incoming sensor rate (50Hz = 20ms per frame)
DT = 1.0 / SAMPLE_HZ      # time step in seconds


# --- MOCK SENSOR DATA GENERATOR ---
def mock_sensor_stream(n_samples=250):
    """
    Simulates a stream of IMU readings (accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z).
    Includes a forward drive phase, followed by a stop (stationary) phase.
    """
    rng = np.random.default_rng(seed=42)
    for i in range(n_samples):
        # First 150 samples: driving forward at ~10 m/s with small sensor noise & bias
        # Last 100 samples: stopped at a red light (zero velocity)
        is_driving = i < 150

        if is_driving:
            forward_accel = 0.5 if i < 30 else 0.0
            accel = np.array([forward_accel + 0.04, 0.0, 9.81]) + rng.normal(0.0, 0.05, size=3)
            gyro = np.array([0.0, 0.0, 0.005]) + rng.normal(0.0, 0.01, size=3)
        else:
            # Stopped at red light
            accel = np.array([0.04, 0.0, 9.81]) + rng.normal(0.0, 0.02, size=3)
            gyro = np.array([0.0, 0.0, 0.005]) + rng.normal(0.0, 0.003, size=3)

        yield np.concatenate([accel, gyro])


# --- SLIDING WINDOW BUFFER ---
class SlidingWindowBuffer:
    """Keeps the last WINDOW_SIZE sensor frames for the model to inspect."""
    def __init__(self, window_size):
        self.buffer = deque(maxlen=window_size)

    def add(self, frame):
        self.buffer.append(frame)

    def is_ready(self):
        return len(self.buffer) == self.buffer.maxlen

    def as_array(self):
        return np.array(self.buffer)


# --- STUB "AI MODEL" ---
def predict_error_stub(window):
    """
    Placeholder for the trained AI model.
    Real version: load a .onnx / .pt file here and run inference on `window`.
    """
    return np.zeros(3)


# --- STAGE 1: NAIVE DOUBLE INTEGRATION ---
class NaiveDeadReckoningState:
    """
    Tracks position/velocity by naive double integration of accelerometer data.
    Suffers from rapid exponential drift due to gravity leakage and sensor biases.
    """
    def __init__(self):
        self.velocity = np.zeros(3)
        self.position = np.zeros(3)
        self.gravity = np.array([0.0, 0.0, 9.81])

    def update(self, accel, correction, dt):
        corrected_accel = accel - correction - self.gravity
        self.velocity += corrected_accel * dt
        self.position += self.velocity * dt
        return self.position.copy()


# --- STAGE 2: PHYSICS-BASED FILTER (ESKF + NHC + ZUPT) ---
class PhysicsDeadReckoningState:
    """
    Stage 2 Filter: 15-state Error-State Kalman Filter.
    Features:
      - Removes gravity via quaternion 3D orientation tracking
      - Enforces Non-Holonomic Constraints (lateral/vertical velocity ~ 0)
      - Detects stationary vehicle state to apply Zero Velocity Updates (ZUPT)
    """
    def __init__(self, sample_hz=50.0):
        self.eskf = ErrorStateKalmanFilter()
        self.detector = StationaryDetector(window_size=int(sample_hz * 0.5))
        self.dt = 1.0 / sample_hz

    def update(self, accel, gyro):
        is_stationary = self.detector.add_sample(accel, gyro)
        self.eskf.predict(accel, gyro, self.dt)

        if is_stationary:
            self.eskf.update_zupt()
        else:
            self.eskf.update_nhc()

        return self.eskf.p.copy(), is_stationary


# --- MAIN RUNNER (SIDE-BY-SIDE DEMO) ---
def run():
    print("=" * 72)
    print(" SIH DEAD RECKONING PROTOTYPE: STAGE 1 vs. STAGE 2 COMPARISON")
    print("=" * 72)

    buffer = SlidingWindowBuffer(WINDOW_SIZE)
    naive_state = NaiveDeadReckoningState()
    physics_state = PhysicsDeadReckoningState(SAMPLE_HZ)

    step_idx = 0
    for frame in mock_sensor_stream(n_samples=250):
        step_idx += 1
        buffer.add(frame)

        if not buffer.is_ready():
            continue  # Warm up buffer

        window = buffer.as_array()
        accel = frame[:3]
        gyro = frame[3:]

        # Run Stage 1: Naive double integration
        correction = predict_error_stub(window)
        naive_pos = naive_state.update(accel, correction, DT)

        # Run Stage 2: Physics-based EKF with NHC & ZUPT
        physics_pos, is_stationary = physics_state.update(accel, gyro)

        # Display progress every 25 frames (~0.5s)
        if step_idx % 25 == 0:
            status = "STOPPED (ZUPT Active)" if is_stationary else "DRIVING (NHC Active)"
            print(f"Frame {step_idx:3d} | [{status:<20}]")
            print(f"   [Stage 1 Naive]   Pos (X, Y, Z): {np.round(naive_pos, 2)}")
            print(f"   [Stage 2 Physics] Pos (X, Y, Z): {np.round(physics_pos, 2)}")
            print("-" * 72)

        # Pace slightly for smooth console inspection
        time.sleep(0.01)

    print("\nSummary:")
    print(f"Final Stage 1 Naive Position (Drifting):   {np.round(naive_pos, 2)}")
    print(f"Final Stage 2 Physics Position (Bounded): {np.round(physics_pos, 2)}")
    print("=" * 72)


if __name__ == "__main__":
    run()
