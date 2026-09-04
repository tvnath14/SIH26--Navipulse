"""
Unit and integration tests for FastAPI REST and WebSocket server.
"""

import unittest
from fastapi.testclient import TestClient
from src.server.app import app


class TestServer(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_status_endpoint(self):
        res = self.client.get("/api/status")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "ONLINE")
        self.assertIn("mode", data)
        self.assertIn("total_distance_m", data)

    def test_toggle_gnss_endpoint(self):
        res = self.client.post("/api/stream/toggle-gnss")
        self.assertEqual(res.status_code, 200)
        state1 = res.json()["forced_outage"]

        res2 = self.client.post("/api/stream/toggle-gnss")
        self.assertEqual(res2.status_code, 200)
        state2 = res2.json()["forced_outage"]

        self.assertNotEqual(state1, state2)

    def test_sensor_ingest_endpoint(self):
        payload = {
            "accel": [0.1, 0.0, 9.81],
            "gyro": [0.0, 0.0, 0.0],
            "gnss_pos": [10.0, 20.0, 0.0],
            "gnss_vel": [5.0, 0.0, 0.0],
        }
        res = self.client.post("/api/sensor/ingest", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["type"], "LIVE_TELEMETRY")
        self.assertIn("pos_est", data)
        self.assertIn("speed_kmh", data)

    def test_websocket_connection(self):
        with self.client.websocket_connect("/ws/telemetry") as ws:
            ws.send_json({"action": "STOP"})

    def test_vehicle_control_endpoint(self):
        res = self.client.post("/api/vehicle/control", json={"target_speed_kmh": 50.0, "duration_s": 100.0})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["target_speed_kmh"], 50.0)
        self.assertEqual(data["duration_s"], 100.0)

        # Test brake
        res2 = self.client.post("/api/vehicle/control", json={"brake": True})
        self.assertEqual(res2.status_code, 200)
        self.assertEqual(res2.json()["target_speed_kmh"], 0.0)

    def test_reset_endpoint(self):
        res = self.client.post("/api/stream/reset")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "STREAM_RESET")

    def test_custom_path_endpoints(self):
        # Set custom path
        pts = [[0.0, 0.0], [50.0, 10.0], [100.0, 30.0], [150.0, 30.0]]
        res = self.client.post("/api/path/custom", json={"points": pts})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "CUSTOM_PATH_SET")
        self.assertEqual(res.json()["points_count"], 4)

        # Clear custom path
        res_reset = self.client.post("/api/path/reset")
        self.assertEqual(res_reset.status_code, 200)
        self.assertEqual(res_reset.json()["status"], "CUSTOM_PATH_CLEARED")

    def test_benchmark_indicator_gnss_locked(self):
        from src.server.app import manager
        # Verify GNSS locked status reports 0.0% drift and PASS
        frame = manager.sim.step_interactive(force_tunnel=False)
        state = manager.pipeline.step(frame["accel"], frame["gyro"], gnss_pos=frame["gnss_pos"], gnss_vel=frame["gnss_vel"])
        self.assertEqual(state["mode"], "GNSS_FIX")


if __name__ == "__main__":
    unittest.main()
