"""
Evaluate accumulated position drift during simulated GPS blackouts.

Run:
py evaluate_blackouts.py "cleaned_data.csv.txt" "idr_comprehensive_model.pkl"
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from model import (
    REQUIRED_COLUMNS,
    infer_sample_rate_hz,
    latlon_to_deltas,
    read_csv,
    normalize_col,
)
from utils import extract_rolling_features


def load_and_process_data(data_path):
    df = read_csv(data_path)
    df.columns = [normalize_col(c) for c in df.columns]

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")

    df = df.rename(columns=REQUIRED_COLUMNS)

    numeric_columns = list(REQUIRED_COLUMNS.values())
    df[numeric_columns] = df[numeric_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    df = df.dropna(subset=numeric_columns).reset_index(drop=True)

    # A timestamp reset starts a new driving recording.
    df["segment_id"] = (
        df["timestamp"].diff().le(0).fillna(False).cumsum()
    )

    sample_rate_hz = infer_sample_rate_hz(
        df["timestamp"],
        df["segment_id"],
    )

    processed_segments = []

    for _, segment in df.groupby("segment_id", sort=False):
        segment = segment.copy()

        # GPS position changes are used only as evaluation ground truth.
        delta_north, delta_east = latlon_to_deltas(
            segment["lat"],
            segment["lon"],
        )

        segment["true_north_m"] = np.nancumsum(
            np.nan_to_num(delta_north, nan=0.0)
        )
        segment["true_east_m"] = np.nancumsum(
            np.nan_to_num(delta_east, nan=0.0)
        )

        window_size = max(2, round(0.5 * sample_rate_hz))

        processed_segment = extract_rolling_features(
            segment,
            window_size=window_size,
            fs=sample_rate_hz,
        )

        processed_segments.append(processed_segment)

    processed_df = pd.concat(
        processed_segments,
        ignore_index=True,
    )

    return processed_df, sample_rate_hz


def evaluate_blackout(
    segment,
    model,
    feature_columns,
    target_columns,
    sample_rate_hz,
    blackout_seconds,
):
    blackout_steps = round(blackout_seconds * sample_rate_hz)

    if len(segment) <= blackout_steps:
        return None

    north_index = target_columns.index("delta_north")
    east_index = target_columns.index("delta_east")

    predictions = model.predict(segment[feature_columns])

    position_errors = []
    stationary_baseline_errors = []

    # Non-overlapping blackout windows.
    for start in range(
        0,
        len(segment) - blackout_steps + 1,
        blackout_steps,
    ):
        end = start + blackout_steps

        # GPS must be reliable at blackout start and end.
        if (
            segment["gps_accuracy"].iloc[start] > 5.0
            or segment["gps_accuracy"].iloc[end] > 5.0
        ):
            continue

        # Sum model-predicted movement during the GPS blackout.
        predicted_delta_ne = predictions[
            start + 1:end + 1,
            [north_index, east_index],
        ].sum(axis=0)

        # Actual GPS movement during the same period.
        true_delta_ne = np.array(
            [
                segment["true_north_m"].iloc[end]
                - segment["true_north_m"].iloc[start],
                segment["true_east_m"].iloc[end]
                - segment["true_east_m"].iloc[start],
            ]
        )

        # Final GPS-denied position error.
        position_error = np.linalg.norm(
            predicted_delta_ne - true_delta_ne
        )

        # Baseline: remain at last known GPS position.
        stationary_error = np.linalg.norm(true_delta_ne)

        position_errors.append(float(position_error))
        stationary_baseline_errors.append(float(stationary_error))

    if not position_errors:
        return None

    return {
        "windows": len(position_errors),
        "median_error_m": float(np.median(position_errors)),
        "mean_error_m": float(np.mean(position_errors)),
        "p95_error_m": float(np.percentile(position_errors, 95)),
        "max_error_m": float(np.max(position_errors)),
        "stationary_baseline_m": float(
            np.median(stationary_baseline_errors)
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "data",
        type=Path,
        help="Path to cleaned_data.csv.txt",
    )

    parser.add_argument(
        "model",
        type=Path,
        help="Path to idr_comprehensive_model.pkl",
    )

    parser.add_argument(
        "--durations",
        default="10,30,60",
        help="Comma-separated blackout durations in seconds",
    )

    args = parser.parse_args()

    with open(args.model, "rb") as file:
        artifact = pickle.load(file)

    required_keys = {"model", "features", "targets"}

    if not required_keys.issubset(artifact.keys()):
        raise ValueError(
            "This is not a compatible idr_comprehensive_model.pkl file."
        )

    if (
        "delta_north" not in artifact["targets"]
        or "delta_east" not in artifact["targets"]
    ):
        raise ValueError(
            "Model must contain delta_north and delta_east targets."
        )

    processed_df, sample_rate_hz = load_and_process_data(args.data)

    # The last segment was the hold-out recording in model.py.
    segment_id = processed_df["segment_id"].max()

    test_segment = processed_df[
        processed_df["segment_id"] == segment_id
    ].reset_index(drop=True)

    durations = [
        float(duration)
        for duration in args.durations.split(",")
    ]

    print(
        f"Evaluating held-out segment {segment_id} "
        f"at {sample_rate_hz:.2f} Hz"
    )

    for blackout_seconds in durations:
        result = evaluate_blackout(
            segment=test_segment,
            model=artifact["model"],
            feature_columns=artifact["features"],
            target_columns=artifact["targets"],
            sample_rate_hz=sample_rate_hz,
            blackout_seconds=blackout_seconds,
        )

        if result is None:
            print(
                f"{blackout_seconds:g}s blackout: "
                "no valid GPS-anchored windows."
            )
            continue

        print(
            f"{blackout_seconds:g}s blackout | "
            f"windows={result['windows']} | "
            f"median end error={result['median_error_m']:.2f} m | "
            f"mean={result['mean_error_m']:.2f} m | "
            f"p95={result['p95_error_m']:.2f} m | "
            f"max={result['max_error_m']:.2f} m | "
            f"stationary baseline={result['stationary_baseline_m']:.2f} m"
        )


if __name__ == "__main__":
    main()