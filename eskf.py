"""
Error-State Extended Kalman Filter (ESKF) for Vehicle Inertial Dead Reckoning.

States (Nominal 16D, Error 15D):
  - Position: p in R^3 (world frame ENU)
  - Velocity: v in R^3 (world frame ENU)
  - Attitude: q in H (Hamiltonian quaternion, body -> world R_wb)
  - Accel Bias: b_a in R^3 (body frame)
  - Gyro Bias:  b_g in R^3 (body frame)

Constraints & Corrections:
  1. Non-Holonomic Constraints (NHC): v_y_body = 0, v_z_body = 0
  2. Zero Velocity Updates (ZUPT): v_world = 0 during vehicle stops
  3. AI Forward Velocity Update: v_x_body = v_AI
  4. GNSS Fix Update: p_world = p_gnss, v_world = v_gnss
"""

from typing import Optional
import numpy as np
from src.kinematics.quaternion import (
    quat_normalize,
    quat_multiply,
    quat_to_rot_matrix,
    rot_vector_to_quat,
    skew_symmetric,
    euler_from_quat,
)


class ErrorStateKalmanFilter:
    """
    15-State ESKF Engine for Real-Time Vehicle Dead Reckoning.
    """

    GRAVITY_MAG = 9.80665  # m/s^2

    def __init__(
        self,
        init_pos: np.ndarray = None,
        init_vel: np.ndarray = None,
        init_quat: np.ndarray = None,
        sigma_accel_noise: float = 0.2,       # m/s^2 / sqrt(Hz)
        sigma_gyro_noise: float = 0.02,        # rad/s / sqrt(Hz)
        sigma_accel_bias: float = 1e-4,        # random walk
        sigma_gyro_bias: float = 1e-5,         # random walk
        r_nhc: float = 0.15,                   # NHC measurement standard deviation (m/s)
        r_zupt: float = 0.05,                  # ZUPT measurement standard deviation (m/s)
        r_ai_speed: float = 0.3,               # AI speed measurement standard deviation (m/s)
    ):
        # Nominal states
        self.p = np.zeros(3) if init_pos is None else np.array(init_pos, dtype=float)
        self.v = np.zeros(3) if init_vel is None else np.array(init_vel, dtype=float)
        self.q = np.array([1.0, 0.0, 0.0, 0.0]) if init_quat is None else quat_normalize(np.array(init_quat, dtype=float))
        self.b_a = np.zeros(3)
        self.b_g = np.zeros(3)

        self.g_w = np.array([0.0, 0.0, self.GRAVITY_MAG])

        # Noise parameters
        self.sigma_acc = sigma_accel_noise
        self.sigma_gyro = sigma_gyro_noise
        self.sigma_ba = sigma_accel_bias
        self.sigma_bg = sigma_gyro_bias

        self.r_nhc = r_nhc
        self.r_zupt = r_zupt
        self.r_ai_speed = r_ai_speed

        # Error state covariance P (15x15)
        # States: [delta_p (3), delta_v (3), delta_theta (3), delta_ba (3), delta_bg (3)]
        self.P = np.diag([
            1.0, 1.0, 1.0,           # pos variance
            0.5, 0.5, 0.5,           # vel variance
            0.01, 0.01, 0.01,        # rot variance (rad^2)
            0.05, 0.05, 0.05,        # accel bias variance
            0.005, 0.005, 0.005,     # gyro bias variance
        ])

    @property
    def rot_matrix(self) -> np.ndarray:
        """Body to World rotation matrix R_wb."""
        return quat_to_rot_matrix(self.q)

    @property
    def body_velocity(self) -> np.ndarray:
        """Velocity in vehicle body frame: v_body = R_wb^T @ v_world."""
        return self.rot_matrix.T @ self.v

    @property
    def euler_angles(self) -> tuple[float, float, float]:
        """(roll, pitch, yaw) in degrees."""
        r, p, y = euler_from_quat(self.q)
        return float(np.degrees(r)), float(np.degrees(p)), float(np.degrees(y))

    def predict(self, accel_raw: np.ndarray, gyro_raw: np.ndarray, dt: float):
        """
        Nominal state strapdown propagation + Error covariance propagation.
        accel_raw: raw accelerometer [ax, ay, az] in m/s^2 (includes gravity)
        gyro_raw:  raw gyroscope [gx, gy, gz] in rad/s
        dt:        sample period in seconds
        """
        # 1. Bias compensation
        w_unbiased = gyro_raw - self.b_g
        a_unbiased = accel_raw - self.b_a

        # 2. Attitude propagation
        delta_theta = w_unbiased * dt
        delta_q = rot_vector_to_quat(delta_theta)
        self.q = quat_normalize(quat_multiply(self.q, delta_q))
        R = self.rot_matrix

        # 3. Acceleration in world frame (subtracting gravity)
        a_world = R @ a_unbiased - self.g_w

        # 4. Position and Velocity propagation
        self.p += self.v * dt + 0.5 * a_world * (dt ** 2)
        self.v += a_world * dt

        # 5. Continuous error-state Jacobian F_x
        # delta_p_dot = delta_v
        # delta_v_dot = -R * [a_unbiased]_x * delta_theta - R * delta_ba
        # delta_theta_dot = -[w_unbiased]_x * delta_theta - delta_bg
        Fx = np.eye(15)
        Fx[0:3, 3:6] = np.eye(3) * dt

        a_skew = skew_symmetric(a_unbiased)
        Fx[3:6, 6:9] = -R @ a_skew * dt
        Fx[3:6, 9:12] = -R * dt

        w_skew = skew_symmetric(w_unbiased)
        Fx[6:9, 6:9] = np.eye(3) - w_skew * dt
        Fx[6:9, 12:15] = -np.eye(3) * dt

        # 6. Discrete process noise Q_d
        Q_d = np.zeros((15, 15))
        Q_d[0:3, 0:3] = np.eye(3) * (self.sigma_acc ** 2) * (dt ** 3) / 3.0
        Q_d[3:6, 3:6] = np.eye(3) * (self.sigma_acc ** 2) * dt
        Q_d[6:9, 6:9] = np.eye(3) * (self.sigma_gyro ** 2) * dt
        Q_d[9:12, 9:12] = np.eye(3) * (self.sigma_ba ** 2) * dt
        Q_d[12:15, 12:15] = np.eye(3) * (self.sigma_bg ** 2) * dt

        # 7. Covariance propagation
        self.P = Fx @ self.P @ Fx.T + Q_d

    def _apply_correction(self, H: np.ndarray, residual: np.ndarray, R_meas: np.ndarray):
        """
        Generic Kalman measurement update and error injection.
        H: Jacobian (m x 15)
        residual: y = z_meas - h(x) (m,)
        R_meas: Measurement covariance (m x m)
        """
        S = H @ self.P @ H.T + R_meas
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ residual

        # Inject error into nominal states
        self.p += dx[0:3]
        self.v += dx[3:6]

        delta_rot = rot_vector_to_quat(dx[6:9])
        self.q = quat_normalize(quat_multiply(self.q, delta_rot))

        self.b_a += dx[9:12]
        self.b_g += dx[12:15]

        # Joseph form covariance update for numerical stability
        I_KH = np.eye(15) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_meas @ K.T

    def update_nhc(self):
        """
        Non-Holonomic Constraints (NHC):
        Vehicles do not slip laterally or bounce vertically on roads:
          v_y_body = 0, v_z_body = 0
        """
        R = self.rot_matrix
        v_b = R.T @ self.v  # [vx_body, vy_body, vz_body]

        # Measurement: z = [0, 0], h(x) = [v_b[1], v_b[2]]
        residual = np.array([-v_b[1], -v_b[2]])

        # Jacobian H (2 x 15)
        H = np.zeros((2, 15))
        # d(v_b) / d(v_w) = R^T
        H[0, 3:6] = R.T[1, :]
        H[1, 3:6] = R.T[2, :]

        # d(v_b) / d(theta_b) = [v_b]_x in body frame
        v_b_skew = skew_symmetric(v_b)
        H[0, 6:9] = v_b_skew[1, :]
        H[1, 6:9] = v_b_skew[2, :]

        R_meas = np.eye(2) * (self.r_nhc ** 2)
        self._apply_correction(H, residual, R_meas)

    def update_zupt(self, gyro_raw: Optional[np.ndarray] = None):
        """
        Zero Velocity Update (ZUPT) and Zero Angular Rate Update (ZARU):
        When vehicle is stationary:
          v_world = [0, 0, 0]
          omega_body = [0, 0, 0] (rapidly calibrates gyro bias b_g)
        """
        if gyro_raw is not None:
            w_unbiased = gyro_raw - self.b_g
            residual = np.concatenate([-self.v, -w_unbiased])
            H = np.zeros((6, 15))
            H[0:3, 3:6] = np.eye(3)
            H[3:6, 12:15] = -np.eye(3)
            R_meas = np.diag([self.r_zupt ** 2] * 3 + [(self.r_zupt * 0.1) ** 2] * 3)
        else:
            residual = -self.v  # 0 - v
            H = np.zeros((3, 15))
            H[0:3, 3:6] = np.eye(3)
            R_meas = np.eye(3) * (self.r_zupt ** 2)

        self._apply_correction(H, residual, R_meas)

    def update_ai_speed(self, ai_forward_speed: float):
        """
        AI Forward Speed Update:
        AI model predicts forward velocity v_x_body.
        """
        R = self.rot_matrix
        v_b = R.T @ self.v
        v_b_x = v_b[0]

        residual = np.array([ai_forward_speed - v_b_x])

        H = np.zeros((1, 15))
        H[0, 3:6] = R.T[0, :]

        # d(v_b_x) / d(theta_b) = [v_b]_x[0, :]
        v_b_skew = skew_symmetric(v_b)
        H[0, 6:9] = v_b_skew[0, :]

        R_meas = np.array([[self.r_ai_speed ** 2]])
        self._apply_correction(H, residual, R_meas)

    def update_gnss(self, gnss_pos: np.ndarray, gnss_vel: np.ndarray = None, pos_std: float = 1.5, vel_std: float = 0.2):
        """
        GNSS Position and Velocity fix update.
        """
        if gnss_vel is not None:
            residual = np.concatenate([gnss_pos - self.p, gnss_vel - self.v])
            H = np.zeros((6, 15))
            H[0:3, 0:3] = np.eye(3)
            H[3:6, 3:6] = np.eye(3)
            R_meas = np.diag([pos_std**2] * 3 + [vel_std**2] * 3)
        else:
            residual = gnss_pos - self.p
            H = np.zeros((3, 15))
            H[0:3, 0:3] = np.eye(3)
            R_meas = np.eye(3) * (pos_std ** 2)

        self._apply_correction(H, residual, R_meas)
