"""
Realistic Ground Vehicle Kinematic Simulator.
Generates ground truth trajectories and synthetically corrupted smartphone IMU data
(incorporating chassis vibrations, road roughness, sensor bias, and noise).
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from src.kinematics.quaternion import quat_from_euler, quat_to_rot_matrix


class VehicleSimulator:
    """
    Simulates a ground vehicle driving through open roads, entering a tunnel,
    stopping at a light, and turning.
    """

    def __init__(
        self,
        sample_hz: float = 50.0,
        accel_bias: np.ndarray = None,
        gyro_bias: np.ndarray = None,
        accel_noise_std: float = 0.08,
        gyro_noise_std: float = 0.008,
        vibration_amplitude: float = 0.12,
        seed: int = 42,
    ):
        self.dt = 1.0 / sample_hz
        self.sample_hz = sample_hz
        self.rng = np.random.default_rng(seed)

        # Realistic smartphone MEMS sensor biases
        self.accel_bias = np.array([0.04, -0.03, 0.05]) if accel_bias is None else accel_bias
        self.gyro_bias = np.array([0.002, -0.003, 0.005]) if gyro_bias is None else gyro_bias

        self.accel_noise_std = accel_noise_std
        self.gyro_noise_std = gyro_noise_std
        self.vibration_amplitude = vibration_amplitude

        # Initialize interactive driving state & custom path
        self.custom_path = None
        self.reset_interactive()

    def reset_interactive(self):
        """Resets the interactive driving simulation to origin or start of custom path."""
        self.t = 0.0
        self.step_idx = 0
        self.speed = 0.0             # m/s
        self.target_speed_mps = 12.0 # default ~43 km/h
        self.forward_accel = 0.0
        self.is_emergency_brake = False
        self.is_stopped = True
        self.pitch = 0.0
        self.roll = 0.0
        self.dist_traveled = 0.0

        if hasattr(self, 'custom_path') and self.custom_path is not None:
            self.pos = np.array([self.custom_path["x"][0], self.custom_path["y"][0], 0.0])
            self.yaw = float(self.custom_path["yaw"][0])
        else:
            self.pos = np.zeros(3)
            self.yaw = 0.0
        self.vel = np.zeros(3)

    def set_custom_path(self, waypoints: list):
        """
        Sets a user-drawn 2D path [(x, y), ...] in meters.
        Interpolates densely and precomputes cumulative distance, heading, and curvature.
        """
        if not waypoints or len(waypoints) < 2:
            self.clear_custom_path()
            return

        pts = np.array(waypoints, dtype=float)
        diffs = np.diff(pts, axis=0)
        seg_lens = np.linalg.norm(diffs, axis=1)

        valid_mask = seg_lens > 1e-4
        if not np.any(valid_mask):
            return

        s_accum = np.insert(np.cumsum(seg_lens), 0, 0.0)
        total_len = float(s_accum[-1])
        if total_len < 1.0:
            return

        # Resample at dense 0.25m resolution for smooth steering kinematics
        num_samples = max(20, int(total_len / 0.25))
        s_eval = np.linspace(0, total_len, num_samples)

        x_interp = np.interp(s_eval, s_accum, pts[:, 0])
        y_interp = np.interp(s_eval, s_accum, pts[:, 1])

        # Smooth waypoints with Gaussian filter to eliminate discrete mouse jitter
        sigma_samples = max(2, int(2.5 / 0.25))  # 2.5m smoothing radius
        x_smooth = gaussian_filter1d(x_interp, sigma=sigma_samples)
        y_smooth = gaussian_filter1d(y_interp, sigma=sigma_samples)

        dx = np.gradient(x_smooth, s_eval)
        dy = np.gradient(y_smooth, s_eval)
        yaws = np.unwrap(np.arctan2(dy, dx))
        yaws_smooth = gaussian_filter1d(yaws, sigma=sigma_samples)
        kappas = np.gradient(yaws_smooth, s_eval)

        # Clamp curvature to physical passenger car limits (R_min ~ 6.7m -> max kappa 0.15 rad/m)
        kappa_max = 0.15
        kappas = np.clip(kappas, -kappa_max, kappa_max)

        self.custom_path = {
            "s": s_eval,
            "x": x_smooth,
            "y": y_smooth,
            "yaw": yaws_smooth,
            "kappa": kappas,
            "total_len": total_len,
            "raw_pts": pts.tolist(),
        }
        self.reset_interactive()

    def clear_custom_path(self):
        """Clears user-defined path and restores default procedural road."""
        self.custom_path = None
        self.reset_interactive()

    def set_target_speed(self, speed_kmh: float):
        """Sets the vehicle's cruise speed target in km/h."""
        self.target_speed_mps = max(0.0, speed_kmh / 3.6)
        if self.target_speed_mps > 0:
            self.is_emergency_brake = False

    def brake(self, emergency: bool = False):
        """Applies realistic service or emergency brakes to bring vehicle to 0 km/h."""
        self.target_speed_mps = 0.0
        self.is_emergency_brake = emergency

    def step_interactive(self, force_tunnel: bool = False) -> dict:
        """
        Executes one realistic physics step (at sample_hz, typically 50Hz = 20ms).
        Simulates acceleration, braking, suspension pitch/roll, steering,
        engine vibration harmonics, and IMU sensor corruption.
        """
        self.t += self.dt
        self.step_idx += 1
        gravity_w = np.array([0.0, 0.0, 9.80665])

        # Acceleration & Braking Dynamics (Passenger Car Kinematics)
        diff = self.target_speed_mps - self.speed
        if self.is_emergency_brake:
            self.forward_accel = -6.5  # Hard emergency stop
            self.speed = max(0.0, self.speed + self.forward_accel * self.dt)
            if self.speed <= 0.05:
                self.speed = 0.0
                self.forward_accel = 0.0
                self.is_stopped = True
                self.is_emergency_brake = False
            else:
                self.is_stopped = False
        elif diff < -0.1:  # Service Braking
            self.forward_accel = -3.5
            self.speed = max(self.target_speed_mps, self.speed + self.forward_accel * self.dt)
            if self.speed <= 0.05:
                self.speed = 0.0
                self.forward_accel = 0.0
                self.is_stopped = True
            else:
                self.is_stopped = False
        elif diff > 0.1:  # Smooth Acceleration
            self.forward_accel = 2.2
            self.speed = min(self.target_speed_mps, self.speed + self.forward_accel * self.dt)
            self.is_stopped = False
        else:  # Cruising at target speed
            self.forward_accel = 0.0
            self.is_stopped = (self.speed <= 0.05)

        # Curvature & Trajectory Dynamics
        if hasattr(self, 'custom_path') and self.custom_path is not None:
            p_info = self.custom_path
            total_len = p_info["total_len"]

            # When reaching the end of a custom path, cleanly stop at destination
            if self.dist_traveled >= total_len:
                self.dist_traveled = total_len
                self.speed = 0.0
                self.forward_accel = 0.0
                self.is_stopped = True

            s_cur = min(self.dist_traveled, total_len)
            s_eval = p_info["s"]

            curr_x = float(np.interp(s_cur, s_eval, p_info["x"]))
            curr_y = float(np.interp(s_cur, s_eval, p_info["y"]))
            curr_yaw = float(np.interp(s_cur, s_eval, p_info["yaw"]))
            curr_kappa = float(np.interp(s_cur, s_eval, p_info["kappa"]))

            self.yaw = curr_yaw
            yaw_rate = curr_kappa * self.speed if not self.is_stopped else 0.0
            centripetal_accel = yaw_rate * self.speed

            self.pitch = 0.0
            self.roll = 0.0
            q_true = quat_from_euler(0.0, 0.0, self.yaw)
            R_true = quat_to_rot_matrix(q_true)

            self.pos = np.array([curr_x, curr_y, 0.0])
            self.vel = R_true @ np.array([self.speed, 0.0, 0.0])
            self.dist_traveled += self.speed * self.dt
        else:
            # Procedural Road Curvature & Steering
            d_mod = self.dist_traveled % 1400.0
            if 150.0 <= d_mod < 260.0:
                kappa = 0.005  # Gentle right curve
            elif 420.0 <= d_mod < 550.0:
                kappa = -0.004  # Gentle left curve
            elif 750.0 <= d_mod < 900.0:
                kappa = 0.007  # Medium right curve
            elif 1100.0 <= d_mod < 1250.0:
                kappa = -0.006  # Left curve
            else:
                kappa = 0.0  # Straight road

            yaw_rate = kappa * self.speed if not self.is_stopped else 0.0
            self.yaw += yaw_rate * self.dt
            centripetal_accel = yaw_rate * self.speed

            self.pitch = 0.0
            self.roll = 0.0
            q_true = quat_from_euler(0.0, 0.0, self.yaw)
            R_true = quat_to_rot_matrix(q_true)

            self.vel = R_true @ np.array([self.speed, 0.0, 0.0])
            self.pos += self.vel * self.dt
            self.dist_traveled += self.speed * self.dt

        a_body = np.array([self.forward_accel if not self.is_stopped else 0.0, centripetal_accel, 0.0])
        accel_specific_force_body = a_body + R_true.T @ gravity_w
        gyro_true_body = np.array([0.0, 0.0, yaw_rate])

        # Dynamic engine vibrations & road roughness
        vib = np.zeros(3)
        if not self.is_stopped:
            engine_rpm_freq = 20.0 + (self.speed / 20.0) * 20.0
            scale_vib = self.vibration_amplitude * min(1.5, max(0.4, self.speed / 10.0))
            vib = np.array([
                np.sin(2 * np.pi * engine_rpm_freq * self.t),
                np.cos(2 * np.pi * engine_rpm_freq * self.t),
                np.sin(4 * np.pi * engine_rpm_freq * self.t),
            ]) * scale_vib
        else:
            # Low idle rumble when stopped
            vib = np.array([np.sin(2 * np.pi * 15.0 * self.t), 0.0, 0.0]) * 0.015

        accel_noise = self.rng.normal(0.0, self.accel_noise_std if not self.is_stopped else 0.02, size=3)
        gyro_noise = self.rng.normal(0.0, self.gyro_noise_std if not self.is_stopped else 0.002, size=3)

        accel_noisy = accel_specific_force_body + self.accel_bias + vib + accel_noise
        gyro_noisy = gyro_true_body + self.gyro_bias + gyro_noise

        gnss_valid = not force_tunnel
        if gnss_valid:
            gnss_pos = self.pos.copy() + self.rng.normal(0.0, 0.5, size=3)
            gnss_vel = self.vel.copy() + self.rng.normal(0.0, 0.1, size=3)
        else:
            gnss_pos = None
            gnss_vel = None

        return {
            "timestamp_s": self.t,
            "accel": accel_noisy,
            "gyro": gyro_noisy,
            "true_pos": self.pos.copy(),
            "true_vel": self.vel.copy(),
            "gnss_pos": gnss_pos,
            "gnss_vel": gnss_vel,
            "gnss_valid": gnss_valid,
            "is_stopped": self.is_stopped,
            "forward_accel": self.forward_accel,
            "speed_kmh": self.speed * 3.6,
            "yaw_deg": float(np.degrees(self.yaw)),
        }

    def generate_scenario(self, duration_s: float = 50.0):
        """
        Yields (t, accel_noisy, gyro_noisy, true_pos, true_vel, is_tunnel, is_stopped)
        Scenario profile:
          0s - 5s:   Gentle acceleration to 12 m/s (~43 km/h), open sky
          5s - 20s:  Cruising inside tunnel (GNSS outage starts at t=5s)
          20s - 30s: Deceleration to stop & stationary at traffic signal inside tunnel
          30s - 40s: Acceleration & 90-degree right turn inside tunnel
          40s - 50s: Cruise at 14 m/s (~50 km/h) and exit tunnel
        """
        total_steps = int(duration_s * self.sample_hz)

        # Ground truth state
        pos = np.zeros(3)
        vel = np.zeros(3)
        speed = 0.0
        yaw = 0.0  # radians

        gravity_w = np.array([0.0, 0.0, 9.80665])

        for step in range(total_steps):
            t = step * self.dt

            # Kinematic profile control
            is_stopped = False
            is_tunnel = t >= 5.0  # GNSS outage after 5s

            forward_accel = 0.0
            yaw_rate = 0.0

            if t < 4.0:
                # Accelerate to 12 m/s
                forward_accel = 3.0
            elif t < 18.0:
                # Cruise at 12 m/s
                forward_accel = 0.0
            elif t < 22.0:
                # Decelerate to 0 m/s
                forward_accel = -3.0
            elif t < 30.0:
                # Stopped at signal
                forward_accel = 0.0
                is_stopped = True
            elif t < 34.0:
                # Accelerate again
                forward_accel = 3.0
            elif t < 38.0:
                # 90-degree turn (0.39 rad/s for 4s = ~1.57 rad)
                forward_accel = 0.0
                yaw_rate = np.pi / 8.0
            else:
                # Cruise to end
                forward_accel = 0.0

            # Update ground truth orientation and velocity
            yaw += yaw_rate * self.dt
            q_true = quat_from_euler(0.0, 0.0, yaw)
            R_true = quat_to_rot_matrix(q_true)

            # Track forward scalar speed
            if is_stopped:
                speed = 0.0
            else:
                speed = max(0.0, speed + forward_accel * self.dt)

            # Body acceleration: forward accel in X, centripetal acceleration (omega x v) in Y
            centripetal_accel = yaw_rate * speed
            a_body = np.array([forward_accel if not is_stopped else 0.0, centripetal_accel, 0.0])

            # World velocity and position
            vel = R_true @ np.array([speed, 0.0, 0.0])
            pos += vel * self.dt

            # True specific force measured by accelerometer in body frame: a_body + R^T @ g_w
            accel_specific_force_body = a_body + R_true.T @ gravity_w
            gyro_true_body = np.array([0.0, 0.0, yaw_rate])

            # Add vehicle engine/chassis vibrations (suppressed when stopped)
            vib = np.zeros(3)
            if not is_stopped:
                engine_rpm_freq = 28.0  # Hz
                vib = np.array([
                    np.sin(2 * np.pi * engine_rpm_freq * t),
                    np.cos(2 * np.pi * engine_rpm_freq * t),
                    np.sin(4 * np.pi * engine_rpm_freq * t),
                ]) * self.vibration_amplitude

            # Sensor noise & bias corruption
            accel_noise = self.rng.normal(0.0, self.accel_noise_std, size=3)
            gyro_noise = self.rng.normal(0.0, self.gyro_noise_std, size=3)

            accel_noisy = accel_specific_force_body + self.accel_bias + vib + accel_noise
            gyro_noisy = gyro_true_body + self.gyro_bias + gyro_noise

            yield (t, accel_noisy, gyro_noisy, pos.copy(), vel.copy(), is_tunnel, is_stopped)
