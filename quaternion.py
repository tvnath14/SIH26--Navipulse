"""
Quaternion and 3D Rotation Kinematics for Inertial Navigation.
Conventions:
  - Hamiltonian quaternion format: q = [w, x, y, z] (w = scalar, [x, y, z] = vector)
  - Rotation matrix R(q) transforms vectors from Body frame to World frame: v_world = R(q) @ v_body
"""

import numpy as np


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion to unit length."""
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Hamiltonian quaternion product q = q1 * q2.
    Both q1, q2 are [w, x, y, z].
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_to_rot_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert unit quaternion q = [w, x, y, z] to a 3x3 Rotation Matrix R_wb (Body to World).
    v_world = R_wb @ v_body
    """
    w, x, y, z = quat_normalize(q)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
        [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)],
    ])


def rot_vector_to_quat(rot_vec: np.ndarray) -> np.ndarray:
    """
    Convert a rotation vector (angle * axis, e.g. delta_theta = omega * dt)
    into a unit quaternion [w, x, y, z].
    """
    angle = np.linalg.norm(rot_vec)
    if angle < 1e-8:
        # First-order Taylor approximation for numerical stability
        w = 1.0 - (angle ** 2) / 8.0
        scale = 0.5 - (angle ** 2) / 48.0
        return quat_normalize(np.array([w, rot_vec[0] * scale, rot_vec[1] * scale, rot_vec[2] * scale]))

    half_angle = 0.5 * angle
    w = np.cos(half_angle)
    scale = np.sin(half_angle) / angle
    return np.array([w, rot_vec[0] * scale, rot_vec[1] * scale, rot_vec[2] * scale])


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """
    Compute 3x3 skew-symmetric matrix [v]_x such that [v]_x @ u = cross(v, u).
    """
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])


def euler_from_quat(q: np.ndarray) -> tuple[float, float, float]:
    """
    Extract Euler angles (roll, pitch, yaw) in radians from quaternion q = [w, x, y, z].
    Sequence: Z-Y-X (yaw, pitch, roll).
    """
    w, x, y, z = quat_normalize(q)

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if np.abs(sinp) >= 1.0:
        pitch = np.copysign(np.pi / 2.0, sinp)
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def quat_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Convert Euler angles (roll, pitch, yaw) in radians to unit quaternion [w, x, y, z].
    """
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z])
