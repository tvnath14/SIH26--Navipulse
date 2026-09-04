"""
Stage 2 Comparative Benchmark: Naive Double-Integration vs. ESKF + NHC + ZUPT.
Tests against the official SIH Performance Benchmark (< 10% drift during GNSS blackout).
"""

import sys
import time
import numpy as np

from src.simulation.vehicle_simulator import VehicleSimulator
from src.pipeline.dead_reckoning_pipeline import DeadReckoningPipeline


class NaiveIntegrator:
    """Stage 1 Baseline: Direct double-integration with gravity removal assumption."""
    def __init__(self):
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.g_w = np.array([0.0, 0.0, 9.80665])

    def step(self, accel: np.ndarray, dt: float, gnss_pos: np.ndarray = None):
        if gnss_pos is not None:
            self.p = gnss_pos.copy()
            return self.p.copy()

        # Naive integration assumes accel minus gravity
        a_corrected = accel - self.g_w
        self.v += a_corrected * dt
        self.p += self.v * dt
        return self.p.copy()


def run_benchmark():
    print("=" * 75)
    print(" SIH INTELLIGENT DEAD RECKONING - STAGE 2 BENCHMARK")
    print(" Scenario: 50s driving simulation (45s GNSS tunnel outage, turn & stop)")
    print("=" * 75)

    sim = VehicleSimulator(sample_hz=50.0, seed=123)
    pipeline = DeadReckoningPipeline(sample_hz=50.0, enable_nhc=True, enable_zupt=True)
    naive = NaiveIntegrator()

    total_steps = 0
    pipeline_latencies = []

    pos_true_history = []
    pos_naive_history = []
    pos_eskf_history = []
    mode_history = []

    t_start_bench = time.perf_counter()

    for t, accel, gyro, true_pos, true_vel, is_tunnel, is_stopped in sim.generate_scenario(duration_s=50.0):
        total_steps += 1

        # Determine GNSS availability (blackout inside tunnel)
        gnss_pos = None if is_tunnel else true_pos
        gnss_vel = None if is_tunnel else true_vel

        # 1. Run Naive Integrator
        naive_pos = naive.step(accel, sim.dt, gnss_pos)

        # 2. Run ESKF Pipeline
        t_step_start = time.perf_counter()
        state = pipeline.step(accel, gyro, gnss_pos=gnss_pos, gnss_vel=gnss_vel)
        t_step_ms = (time.perf_counter() - t_step_start) * 1000.0

        pipeline_latencies.append(t_step_ms)
        pos_true_history.append(true_pos)
        pos_naive_history.append(naive_pos)
        pos_eskf_history.append(state["position"])
        mode_history.append(state["mode"])

        # Periodic logging every 10 seconds of simulated time (500 steps)
        if total_steps % 500 == 0:
            drift_eskf = np.linalg.norm(state["position"] - true_pos)
            drift_naive = np.linalg.norm(naive_pos - true_pos)
            pct_eskf = (drift_eskf / max(1.0, pipeline.total_distance)) * 100.0
            print(
                f"[t={t:4.1f}s | Mode: {state['mode']:<15}] "
                f"Dist: {pipeline.total_distance:5.1f}m | "
                f"ESKF Drift: {drift_eskf:5.2f}m ({pct_eskf:4.1f}%) | "
                f"Naive Drift: {drift_naive:7.1f}m | "
                f"Step: {t_step_ms:.2f}ms"
            )

    total_bench_duration = time.perf_counter() - t_start_bench
    final_true_pos = pos_true_history[-1]
    final_naive_pos = pos_naive_history[-1]
    final_eskf_pos = pos_eskf_history[-1]

    naive_error = float(np.linalg.norm(final_naive_pos - final_true_pos))
    eskf_error = float(np.linalg.norm(final_eskf_pos - final_true_pos))

    total_dist = pipeline.total_distance
    naive_drift_pct = (naive_error / max(1.0, total_dist)) * 100.0
    eskf_drift_pct = (eskf_error / max(1.0, total_dist)) * 100.0

    avg_latency = float(np.mean(pipeline_latencies))
    p95_latency = float(np.percentile(pipeline_latencies, 95))
    max_latency = float(np.max(pipeline_latencies))

    print("\n" + "=" * 75)
    print(" FINAL PERFORMANCE BENCHMARK REPORT")
    print("=" * 75)
    print(f"Total Trajectory Distance:      {total_dist:.2f} meters")
    print(f"GNSS Tunnel Outage Duration:    45.0 seconds (90% of entire run)")
    print(f"Total Samples Processed:        {total_steps} frames")
    print(f"Simulation Benchmark Runtime:   {total_bench_duration:.3f} seconds ({total_steps / total_bench_duration:.1f}x real-time)")
    print("-" * 75)
    print(f"{'Metric':<30} | {'Naive Baseline':<18} | {'Stage 2 ESKF + NHC + ZUPT':<20}")
    print("-" * 75)
    print(f"{'Final Position Error (m)':<30} | {naive_error:18.2f} | {eskf_error:20.2f}")
    print(f"{'Drift Percentage of Distance':<30} | {naive_drift_pct:17.1f}% | {eskf_drift_pct:19.2f}%")
    print(f"{'Average Processing Latency':<30} | {'< 0.1 ms':<18} | {avg_latency:17.3f} ms")
    print(f"{'95th Percentile Latency':<30} | {'< 0.1 ms':<18} | {p95_latency:17.3f} ms")
    print(f"{'Max Processing Latency':<30} | {'< 0.1 ms':<18} | {max_latency:17.3f} ms")
    print("-" * 75)

    sih_target_met = eskf_drift_pct < 10.0
    speed_target_met = max_latency < 20.0

    print(f"SIH Accuracy Target (<10% Drift):      {'PASSED' if sih_target_met else 'FAILED'} ({eskf_drift_pct:.2f}%)")
    print(f"50Hz Real-Time Target (<20ms Latency):  {'PASSED' if speed_target_met else 'FAILED'} (Avg: {avg_latency:.3f}ms, Max: {max_latency:.3f}ms)")
    print("=" * 75)

    return sih_target_met and speed_target_met


if __name__ == "__main__":
    success = run_benchmark()
    sys.exit(0 if success else 1)
