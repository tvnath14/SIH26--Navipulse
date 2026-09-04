import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import sys
sys.path.insert(0, 'AI Model')
from utils import _causal_lowpass

def _find_col(df, *must_contain):
    matches = [c for c in df.columns if all(tok in c for tok in must_contain)]
    if not matches:
        raise KeyError(f"No column found containing {must_contain}. Available columns: {list(df.columns)}")
    return matches[0]

# 1. Read dataset
csv_path = Path('AI Model/cleaned_data.csv')
df = pd.read_csv(csv_path, encoding='latin-1')
df.columns = df.columns.str.strip()

# Find columns dynamically
speed_col = _find_col(df, 'GPS', 'SPEED')
gps_yaw_col = _find_col(df, 'GPS', 'ORIENT')
phone_yaw_col = _find_col(df, 'ORIENTATION (Yaw)')
time_col = _find_col(df, 'TIME')
acc_col = _find_col(df, 'GPS ACCURACY')
lat_col = _find_col(df, 'LATITUDE')
lon_col = _find_col(df, 'LONGITUDE')

raw_axes = [
    _find_col(df, 'ACCELEROMETER X'),
    _find_col(df, 'ACCELEROMETER Y'),
    _find_col(df, 'ACCELEROMETER Z'),
    _find_col(df, 'GYROSCOPE Roll'),
    _find_col(df, 'GYROSCOPE Pitch'),
    _find_col(df, 'GYROSCOPE Yaw'),
]

rename_map = {
    raw_axes[0]: "accel_x", raw_axes[1]: "accel_y", raw_axes[2]: "accel_z",
    raw_axes[3]: "gyro_x", raw_axes[4]: "gyro_y", raw_axes[5]: "gyro_z",
    lat_col: "lat", lon_col: "lon", speed_col: "speed",
    acc_col: "gps_accuracy", time_col: "timestamp",
    phone_yaw_col: "phone_yaw", gps_yaw_col: "gps_heading",
}
df = df.rename(columns=rename_map)

numeric = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z", "lat", "lon", "speed", "gps_accuracy", "timestamp", "phone_yaw", "gps_heading"]
df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
df = df.dropna(subset=numeric).reset_index(drop=True)

df["segment_id"] = df["timestamp"].diff().le(0).fillna(False).cumsum()
fs_hz = 10.0
dt = 1.0 / fs_hz

print(f"Dataset has {len(df)} rows across {df['segment_id'].nunique()} segments.")

# Add phone orientation trigonometric features (causal from magnetometer/IMU)
df["cos_yaw"] = np.cos(np.radians(df["phone_yaw"]))
df["sin_yaw"] = np.sin(np.radians(df["phone_yaw"]))

# Compute rolling features
df["accel_mag"] = np.linalg.norm(df[["accel_x", "accel_y", "accel_z"]], axis=1)
df["gyro_mag"] = np.linalg.norm(df[["gyro_x", "gyro_y", "gyro_z"]], axis=1)

feature_sources = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z", "accel_mag", "gyro_mag"]
window_size = 5

for col in feature_sources:
    filtered = _causal_lowpass(df[col].to_numpy(), cutoff_hz=2.5, fs_hz=fs_hz)
    df[f"{col}_filtered"] = filtered
    roll = pd.Series(filtered, index=df.index).rolling(window=window_size, min_periods=1)
    df[f"{col}_mean"] = roll.mean()
    df[f"{col}_std"] = roll.std(ddof=0)
    df[f"{col}_delta"] = pd.Series(filtered, index=df.index).diff().fillna(0.0)

# True clean targets (forward movement decomposed into North and East using heading):
speed_mps = np.maximum(0.0, df["speed"].to_numpy(float)) / 3.6
gps_h_rad = np.radians(df["gps_heading"].to_numpy(float))

df["delta_north"] = speed_mps * np.cos(gps_h_rad) * dt
df["delta_east"] = speed_mps * np.sin(gps_h_rad) * dt

feature_columns = [c for c in df.columns if c.endswith(("_filtered", "_mean", "_std", "_delta"))] + ["cos_yaw", "sin_yaw"]
targets = ["speed", "delta_north", "delta_east"]

segment_ids = df["segment_id"].to_numpy()
last_seg_id = np.max(segment_ids)
train_mask = segment_ids != last_seg_id

X_train = df.loc[train_mask, feature_columns]
y_train = df.loc[train_mask, targets]

print(f"Training Random Forest on {len(X_train)} samples with {len(feature_columns)} features...")
rf = RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=5, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Save artifact
out_path = Path("AI Model/idr_comprehensive_model.pkl")
artifact = {
    "model": rf,
    "features": feature_columns,
    "targets": targets,
    "sample_rate_hz": fs_hz,
    "window_samples": window_size,
    "units": {"speed": "km/h", "delta_north": "m", "delta_east": "m"}
}
with out_path.open("wb") as f:
    pickle.dump(artifact, f)

print(f"Model successfully saved to {out_path.resolve()}!")