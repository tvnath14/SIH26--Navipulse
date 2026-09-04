"""
IO-VNBD Dataset Parser and Streaming Replayer.
Handles loading, parsing, coordinate conversion (WGS-84 to Local ENU),
and real-time streaming of IMU and GNSS benchmark traces.
"""

import os
import time
import csv
from typing import Iterator, Optional, Tuple, Dict, Any
import numpy as np

# Earth radius in meters (WGS-84 equatorial)
EARTH_RADIUS = 6378137.0


def lat_lon_to_local_enu(
    lat: float, lon: float, alt: float,
    lat0: float, lon0: float, alt0: float,
) -> np.ndarray:
    """
    Projects WGS-84 Geodetic coordinates (lat, lon, alt) to Local Cartesian
    East-North-Up (ENU) coordinates in meters relative to reference (lat0, lon0, alt0).
    """
    d_lat = np.radians(lat - lat0)
    d_lon = np.radians(lon - lon0)
    lat0_rad = np.radians(lat0)

    x_east = d_lon * np.cos(lat0_rad) * EARTH_RADIUS
    y_north = d_lat * EARTH_RADIUS
    z_up = alt - alt0

    return np.array([x_east, y_north, z_up])


def local_enu_to_lat_lon(
    x_east: float, y_north: float, z_up: float,
    lat0: float, lon0: float, alt0: float,
) -> Tuple[float, float, float]:
    """
    Converts Local Cartesian ENU meters back to WGS-84 (lat, lon, alt).
    """
    lat0_rad = np.radians(lat0)
    d_lat = np.degrees(y_north / EARTH_RADIUS)
    d_lon = np.degrees(x_east / (EARTH_RADIUS * np.cos(lat0_rad)))

    return lat0 + d_lat, lon0 + d_lon, alt0 + z_up


def create_sample_iovnbd_csv(
    filepath: str,
    duration_s: float = 50.0,
    sample_hz: float = 50.0,
    lat0: float = 28.6139, # New Delhi coordinates reference
    lon0: float = 77.2090,
    alt0: float = 216.0,
) -> str:
    """
    Generates a realistic IO-VNBD format CSV file for testing and development.
    Includes open sky drive -> underground tunnel entry -> traffic signal stop -> 90 deg turn.
    """
    from src.simulation.vehicle_simulator import VehicleSimulator

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    sim = VehicleSimulator(sample_hz=sample_hz, seed=42)

    fieldnames = [
        "timestamp_s",
        "accel_x", "accel_y", "accel_z",
        "gyro_x", "gyro_y", "gyro_z",
        "gnss_valid",
        "gnss_lat", "gnss_lon", "gnss_alt",
        "true_pos_x", "true_pos_y", "true_pos_z",
        "true_vel_x", "true_vel_y", "true_vel_z",
        "is_stopped",
    ]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for t, accel, gyro, true_pos, true_vel, is_tunnel, is_stopped in sim.generate_scenario(duration_s=duration_s):
            gnss_valid = not is_tunnel
            if gnss_valid:
                lat, lon, alt = local_enu_to_lat_lon(true_pos[0], true_pos[1], true_pos[2], lat0, lon0, alt0)
                # Add tiny GPS measurement noise (~1m)
                rng = np.random.default_rng(int(t * 100))
                lat += rng.normal(0, 1e-5)
                lon += rng.normal(0, 1e-5)
                alt += rng.normal(0, 0.5)
            else:
                lat, lon, alt = "", "", ""

            writer.writerow({
                "timestamp_s": f"{t:.4f}",
                "accel_x": f"{accel[0]:.5f}",
                "accel_y": f"{accel[1]:.5f}",
                "accel_z": f"{accel[2]:.5f}",
                "gyro_x": f"{gyro[0]:.5f}",
                "gyro_y": f"{gyro[1]:.5f}",
                "gyro_z": f"{gyro[2]:.5f}",
                "gnss_valid": 1 if gnss_valid else 0,
                "gnss_lat": f"{lat:.7f}" if gnss_valid else "",
                "gnss_lon": f"{lon:.7f}" if gnss_valid else "",
                "gnss_alt": f"{alt:.2f}" if gnss_valid else "",
                "true_pos_x": f"{true_pos[0]:.4f}",
                "true_pos_y": f"{true_pos[1]:.4f}",
                "true_pos_z": f"{true_pos[2]:.4f}",
                "true_vel_x": f"{true_vel[0]:.4f}",
                "true_vel_y": f"{true_vel[1]:.4f}",
                "true_vel_z": f"{true_vel[2]:.4f}",
                "is_stopped": 1 if is_stopped else 0,
            })

    return filepath


class IOVNBDDataset:
    """
    Parser and streaming provider for IO-VNBD format datasets.
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        if not os.path.exists(filepath):
            # Auto-generate sample benchmark file if missing
            create_sample_iovnbd_csv(filepath)

        self.rows = []
        self.ref_lat = None
        self.ref_lon = None
        self.ref_alt = None
        self._load()

    def _load(self):
        with open(self.filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Find first valid GPS fix for local ENU origin
                if int(row["gnss_valid"]) == 1 and self.ref_lat is None:
                    self.ref_lat = float(row["gnss_lat"])
                    self.ref_lon = float(row["gnss_lon"])
                    self.ref_alt = float(row["gnss_alt"])
                self.rows.append(row)

        if self.ref_lat is None:
            self.ref_lat, self.ref_lon, self.ref_alt = 0.0, 0.0, 0.0

    def __len__(self) -> int:
        return len(self.rows)

    def stream(self, playback_speed: float = 1.0) -> Iterator[Dict[str, Any]]:
        """
        Streams dataset frames one by one.
        playback_speed:
          1.0 = real-time (50Hz = 20ms pause)
          2.0 = 2x speed
          0.0 = batch / instantaneous
        """
        prev_time = None

        for row in self.rows:
            t = float(row["timestamp_s"])

            if playback_speed > 0 and prev_time is not None:
                dt = t - prev_time
                if dt > 0:
                    time.sleep(dt / playback_speed)
            prev_time = t

            accel = np.array([float(row["accel_x"]), float(row["accel_y"]), float(row["accel_z"])])
            gyro = np.array([float(row["gyro_x"]), float(row["gyro_y"]), float(row["gyro_z"])])

            gnss_valid = int(row["gnss_valid"]) == 1
            gnss_pos = None
            gnss_vel = None

            if gnss_valid and row["gnss_lat"] != "":
                lat = float(row["gnss_lat"])
                lon = float(row["gnss_lon"])
                alt = float(row["gnss_alt"])
                gnss_pos = lat_lon_to_local_enu(lat, lon, alt, self.ref_lat, self.ref_lon, self.ref_alt)

            true_pos = np.array([float(row["true_pos_x"]), float(row["true_pos_y"]), float(row["true_pos_z"])])
            true_vel = np.array([float(row["true_vel_x"]), float(row["true_vel_y"]), float(row["true_vel_z"])])

            yield {
                "timestamp_s": t,
                "accel": accel,
                "gyro": gyro,
                "gnss_valid": gnss_valid,
                "gnss_pos": gnss_pos,
                "gnss_vel": gnss_vel,
                "true_pos": true_pos,
                "true_vel": true_vel,
                "is_stopped": int(row["is_stopped"]) == 1,
            }
