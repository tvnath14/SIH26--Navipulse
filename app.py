"""
FastAPI Server & WebSocket Bridge for Live Telemetry and UI Integration.
Provides real-time Dead Reckoning telemetry feeds, interactive vehicle driving controls,
GNSS outage injection, and phone sensor ingestion.
"""

import os
import json
import asyncio
from typing import Set, Optional, Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np

from src.pipeline.dead_reckoning_pipeline import DeadReckoningPipeline, NavigationMode
from src.simulation.vehicle_simulator import VehicleSimulator
from src.data.iovnbd_loader import IOVNBDDataset, create_sample_iovnbd_csv, local_enu_to_lat_lon

app = FastAPI(
    title="SIH Intelligent Dead Reckoning API",
    description="Real-time backend engine for GNSS-denied inertial navigation with interactive driving controls",
    version="2.1.0",
)

# Enable CORS for frontend web app communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
DEFAULT_DATASET = os.path.join(os.path.dirname(__file__), "..", "..", "data", "sample_iovnbd_tunnel_drive.csv")


class SensorIngestPayload(BaseModel):
    """Payload for live smartphone IMU streaming."""
    accel: list[float]  # [ax, ay, az] in m/s^2
    gyro: list[float]   # [gx, gy, gz] in rad/s
    gnss_pos: Optional[list[float]] = None  # [x, y, z] or None if unavailable
    gnss_vel: Optional[list[float]] = None  # [vx, vy, vz] or None


class StreamControlPayload(BaseModel):
    dataset_path: Optional[str] = None
    playback_speed: float = 1.0
    duration_s: Optional[float] = None


class VehicleControlPayload(BaseModel):
    target_speed_kmh: Optional[float] = None
    brake: Optional[bool] = False
    emergency_brake: Optional[bool] = False
    duration_s: Optional[float] = None


class CustomPathPayload(BaseModel):
    points: list[list[float]]  # List of [x, y] waypoints in meters


class StreamingManager:
    """Manages interactive vehicle physics, playback tasks, WebSocket clients, and pipeline states."""

    def __init__(self):
        self.clients: Set[WebSocket] = set()
        self.pipeline = DeadReckoningPipeline(sample_hz=50.0, enable_nhc=True, enable_zupt=True)
        self.sim = VehicleSimulator(sample_hz=50.0, seed=42)
        self.is_streaming = False
        self.force_gnss_outage = False
        self.stream_task: Optional[asyncio.Task] = None
        self.current_step = 0
        self.playback_speed = 1.0
        self.scenario_duration_s = 50.0  # 50s, 100s, 200s, or -1 for Infinite
        self.dataset_path = os.path.abspath(DEFAULT_DATASET)

        # Reference coordinates for WGS-84 projection (New Delhi benchmark coordinates)
        self.ref_lat = 28.6139
        self.ref_lon = 77.2090
        self.ref_alt = 216.0

        # Ensure sample dataset exists
        if not os.path.exists(self.dataset_path):
            create_sample_iovnbd_csv(self.dataset_path, duration_s=50.0)

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.clients.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.clients.discard(websocket)

    async def broadcast(self, packet: Dict[str, Any]):
        if not self.clients:
            return
        dead_clients = set()
        msg_str = json.dumps(packet)
        for client in list(self.clients):
            try:
                await client.send_text(msg_str)
            except Exception:
                dead_clients.add(client)
        for client in dead_clients:
            self.clients.discard(client)

    async def reset(self):
        """Resets the simulator and ESKF pipeline to the starting line."""
        self.sim.reset_interactive()
        if self.sim.custom_path is not None:
            init_pos = np.array([self.sim.custom_path["x"][0], self.sim.custom_path["y"][0], 0.0])
            init_yaw = float(self.sim.custom_path["yaw"][0])
            self.pipeline = DeadReckoningPipeline(
                sample_hz=50.0,
                enable_nhc=True,
                enable_zupt=True,
                init_pos=init_pos,
                init_yaw=init_yaw,
            )
        else:
            self.pipeline = DeadReckoningPipeline(sample_hz=50.0, enable_nhc=True, enable_zupt=True)
        self.current_step = 0
        self.force_gnss_outage = False
        await self.broadcast({"type": "RESET"})

    def set_target_speed(self, speed_kmh: float):
        self.sim.set_target_speed(speed_kmh)

    def brake(self, emergency: bool = False):
        self.sim.brake(emergency)

    def set_duration(self, duration_s: float):
        self.scenario_duration_s = float(duration_s)

    def set_custom_path(self, waypoints: list):
        self.sim.set_custom_path(waypoints)
        if self.sim.custom_path is not None:
            init_pos = np.array([self.sim.custom_path["x"][0], self.sim.custom_path["y"][0], 0.0])
            init_yaw = float(self.sim.custom_path["yaw"][0])
            self.pipeline = DeadReckoningPipeline(
                sample_hz=50.0,
                enable_nhc=True,
                enable_zupt=True,
                init_pos=init_pos,
                init_yaw=init_yaw,
            )
        else:
            self.pipeline = DeadReckoningPipeline(sample_hz=50.0, enable_nhc=True, enable_zupt=True)
        self.current_step = 0

    def clear_custom_path(self):
        self.sim.clear_custom_path()
        self.pipeline = DeadReckoningPipeline(sample_hz=50.0, enable_nhc=True, enable_zupt=True)
        self.current_step = 0

    async def run_stream_loop(self):
        """
        Asynchronous 50Hz streaming loop with live vehicle physics,
        outage simulation, and SIH Dead Reckoning verification.
        """
        base_dt = 1.0 / 50.0

        while self.is_streaming:
            self.current_step += 1

            # Check loop duration limit (if duration_s > 0)
            if self.scenario_duration_s > 0 and self.sim.t >= self.scenario_duration_s:
                await self.broadcast({"type": "LOOP_RESTART"})
                await self.reset()
                await asyncio.sleep(0.5)
                continue

            # Execute realistic vehicle kinematics step
            frame = self.sim.step_interactive(force_tunnel=self.force_gnss_outage)

            accel = frame["accel"]
            gyro = frame["gyro"]
            true_pos = frame["true_pos"]
            gnss_pos = frame["gnss_pos"]
            gnss_vel = frame["gnss_vel"]

            # Run Dead Reckoning Pipeline
            state = self.pipeline.step(accel, gyro, gnss_pos=gnss_pos, gnss_vel=gnss_vel)
            pos_est = state["position"]

            # Compute error & drift metrics
            drift_m = float(np.linalg.norm(pos_est - true_pos))
            speed_kmh = float(np.linalg.norm(state["velocity"])) * 3.6
            outage_dist = state["outage_distance_m"]

            if gnss_pos is not None:
                gnss_status = "LOCKED (GNSS)"
                drift_pct = 0.0
                benchmark_status = "PASS"
            else:
                gnss_status = "BLACKOUT (OUTAGE)"
                # SIH Benchmark: Drift must be <10% of the distance traveled during GNSS blackout
                # Over initial handover / stationary travel (< 20m), benchmark requires absolute error < 2.0m
                if outage_dist >= 20.0:
                    drift_pct = (drift_m / outage_dist) * 100.0
                else:
                    drift_pct = (drift_m / 20.0) * 100.0

                if drift_pct >= 10.0:
                    benchmark_status = "VIOLATED"
                elif drift_pct >= 7.5:
                    benchmark_status = "WARNING"
                else:
                    benchmark_status = "PASS"

            # Geodetic coordinate conversion (WGS-84) for real map display
            lat_est, lon_est, _ = local_enu_to_lat_lon(
                pos_est[0], pos_est[1], pos_est[2],
                self.ref_lat, self.ref_lon, self.ref_alt
            )
            lat_true, lon_true, _ = local_enu_to_lat_lon(
                true_pos[0], true_pos[1], true_pos[2],
                self.ref_lat, self.ref_lon, self.ref_alt
            )

            packet = {
                "type": "TELEMETRY",
                "step": self.current_step,
                "timestamp_s": round(frame["timestamp_s"], 2),
                "mode": state["mode"],
                "gnss_status": gnss_status,
                "speed_kmh": round(speed_kmh, 1),
                "target_speed_kmh": round(self.sim.target_speed_mps * 3.6, 1),
                "accel_ms2": round(float(frame["forward_accel"]), 2),
                "heading_deg": round(state["heading_deg"], 1),
                "roll_deg": round(state["roll_deg"], 1),
                "pitch_deg": round(state["pitch_deg"], 1),
                "pos_est": [round(float(x), 2) for x in pos_est],
                "pos_true": [round(float(x), 2) for x in true_pos],
                "lat_est": round(lat_est, 7),
                "lon_est": round(lon_est, 7),
                "lat_true": round(lat_true, 7),
                "lon_true": round(lon_true, 7),
                "drift_m": round(drift_m, 2),
                "drift_pct": round(drift_pct, 2),
                "benchmark_status": benchmark_status,
                "total_distance_m": round(state["total_distance_m"], 1),
                "outage_distance_m": round(outage_dist, 1),
                "outage_duration_s": round(state["outage_duration_s"], 1),
                "scenario_duration_s": self.scenario_duration_s,
                "latency_ms": round(state["latency_ms"], 2),
                "is_stationary": bool(state["is_stationary"]),
                "forced_outage": self.force_gnss_outage,
            }

            await self.broadcast(packet)

            # 50Hz timing control
            delay = base_dt / max(0.1, self.playback_speed)
            await asyncio.sleep(delay)


manager = StreamingManager()


# --- REST ENDPOINTS ---

@app.get("/api/status")
async def get_status():
    return {
        "status": "ONLINE",
        "is_streaming": manager.is_streaming,
        "forced_outage": manager.force_gnss_outage,
        "connected_clients": len(manager.clients),
        "current_step": manager.current_step,
        "mode": manager.pipeline.current_mode.value,
        "speed_kmh": round(manager.sim.speed * 3.6, 1),
        "target_speed_kmh": round(manager.sim.target_speed_mps * 3.6, 1),
        "is_stopped": manager.sim.is_stopped,
        "total_distance_m": round(manager.pipeline.total_distance, 2),
        "scenario_duration_s": manager.scenario_duration_s,
    }


@app.post("/api/stream/start")
async def start_stream(control: Optional[StreamControlPayload] = None):
    if control:
        if control.dataset_path:
            manager.dataset_path = control.dataset_path
        manager.playback_speed = control.playback_speed
        if control.duration_s is not None:
            manager.set_duration(control.duration_s)

    if not manager.is_streaming:
        manager.is_streaming = True
        manager.stream_task = asyncio.create_task(manager.run_stream_loop())

    return {"status": "STREAMING_STARTED", "speed": manager.playback_speed}


@app.post("/api/stream/stop")
async def stop_stream():
    manager.is_streaming = False
    if manager.stream_task:
        manager.stream_task.cancel()
        manager.stream_task = None
    return {"status": "STREAMING_STOPPED"}


@app.post("/api/stream/reset")
async def reset_stream():
    await manager.reset()
    return {"status": "STREAM_RESET"}


@app.post("/api/stream/toggle-gnss")
async def toggle_gnss():
    manager.force_gnss_outage = not manager.force_gnss_outage
    return {"forced_outage": manager.force_gnss_outage}


@app.post("/api/vehicle/control")
async def control_vehicle(payload: VehicleControlPayload):
    if payload.emergency_brake:
        manager.brake(emergency=True)
    elif payload.brake:
        manager.brake(emergency=False)
    elif payload.target_speed_kmh is not None:
        manager.set_target_speed(payload.target_speed_kmh)

    if payload.duration_s is not None:
        manager.set_duration(payload.duration_s)

    return {
        "target_speed_kmh": round(manager.sim.target_speed_mps * 3.6, 1),
        "is_stopped": manager.sim.is_stopped,
        "duration_s": manager.scenario_duration_s,
    }


@app.post("/api/path/custom")
async def set_custom_path_endpoint(payload: CustomPathPayload):
    manager.set_custom_path(payload.points)
    lat_est, lon_est, _ = local_enu_to_lat_lon(
        manager.sim.pos[0], manager.sim.pos[1], manager.sim.pos[2],
        manager.ref_lat, manager.ref_lon, manager.ref_alt
    )
    init_packet = {
        "type": "TELEMETRY",
        "step": 0,
        "timestamp_s": 0.0,
        "mode": "DEAD_RECKONING",
        "gnss_status": "LOCKED (GNSS)",
        "speed_kmh": 0.0,
        "target_speed_kmh": round(manager.sim.target_speed_mps * 3.6, 1),
        "accel_ms2": 0.0,
        "heading_deg": round(float(np.degrees(manager.sim.yaw)), 1),
        "roll_deg": 0.0,
        "pitch_deg": 0.0,
        "pos_est": [round(float(x), 2) for x in manager.sim.pos],
        "pos_true": [round(float(x), 2) for x in manager.sim.pos],
        "lat_est": round(lat_est, 7),
        "lon_est": round(lon_est, 7),
        "lat_true": round(lat_est, 7),
        "lon_true": round(lon_est, 7),
        "drift_m": 0.0,
        "drift_pct": 0.0,
        "benchmark_status": "PASS",
        "total_distance_m": 0.0,
        "outage_distance_m": 0.0,
        "outage_duration_s": 0.0,
        "scenario_duration_s": manager.scenario_duration_s,
        "latency_ms": 0.0,
        "is_stationary": True,
        "forced_outage": manager.force_gnss_outage,
    }
    await manager.broadcast({"type": "CUSTOM_PATH_SET", "points": payload.points})
    await manager.broadcast(init_packet)
    return {"status": "CUSTOM_PATH_SET", "points_count": len(payload.points)}


@app.post("/api/path/reset")
async def reset_custom_path_endpoint():
    manager.clear_custom_path()
    await manager.broadcast({"type": "CUSTOM_PATH_CLEARED"})
    return {"status": "CUSTOM_PATH_CLEARED"}


@app.post("/api/sensor/ingest")
async def ingest_phone_sensor(payload: SensorIngestPayload):
    """
    Ingest live IMU telemetry from a smartphone application or external device.
    """
    accel = np.array(payload.accel)
    gyro = np.array(payload.gyro)
    gnss_pos = np.array(payload.gnss_pos) if payload.gnss_pos else None
    gnss_vel = np.array(payload.gnss_vel) if payload.gnss_vel else None

    state = manager.pipeline.step(accel, gyro, gnss_pos=gnss_pos, gnss_vel=gnss_vel)
    packet = {
        "type": "LIVE_TELEMETRY",
        "mode": state["mode"],
        "pos_est": [round(float(x), 2) for x in state["position"]],
        "speed_kmh": round(float(np.linalg.norm(state["velocity"])) * 3.6, 1),
        "heading_deg": round(state["heading_deg"], 1),
        "latency_ms": round(state["latency_ms"], 2),
    }
    await manager.broadcast(packet)
    return packet


# --- WEBSOCKET ENDPOINT ---

@app.websocket("/ws/telemetry")
async def websocket_telemetry_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                cmd = json.loads(data)
                action = cmd.get("action")
                if action == "START":
                    await start_stream()
                elif action == "STOP":
                    await stop_stream()
                elif action == "RESET":
                    await reset_stream()
                elif action == "TOGGLE_GNSS":
                    await toggle_gnss()
                elif action == "SET_SPEED":
                    manager.set_target_speed(float(cmd.get("speed_kmh", 0.0)))
                elif action == "BRAKE":
                    manager.brake(emergency=cmd.get("emergency", False))
                elif action == "SET_DURATION":
                    manager.set_duration(float(cmd.get("duration_s", 50.0)))
                elif action == "SET_CUSTOM_PATH":
                    manager.set_custom_path(cmd.get("points", []))
                    await manager.broadcast({"type": "CUSTOM_PATH_SET", "points": cmd.get("points", [])})
                elif action == "CLEAR_CUSTOM_PATH":
                    manager.clear_custom_path()
                    await manager.broadcast({"type": "CUSTOM_PATH_CLEARED"})
            except Exception:
                pass
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# --- DASHBOARD SERVING ---

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def serve_dashboard():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "SIH Dead Reckoning Server Running. Visit /docs for Swagger API."}
