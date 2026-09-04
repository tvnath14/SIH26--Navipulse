import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter

def butter_lowpass_filter(data, cutoff=2.5, fs=10.0, order=4):
    """
    Applies a Butterworth low-pass filter to eliminate high-frequency 
    vibrations (engine idle, potholes) while retaining true vehicle dynamics.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y

def extract_rolling_features(df, window_size=5):
    """
    Computes rolling statistics (mean, standard deviation, magnitude) 
    over a sliding time window to give the model temporal context.
    """
    print("Extracting advanced temporal and magnitude features...")
    
    # Calculate acceleration and gyroscope vector magnitudes
    df['accel_mag'] = np.sqrt(df['accel_x']**2 + df['accel_y']**2 + df['accel_z']**2)
    df['gyro_mag'] = np.sqrt(df['gyro_x']**2 + df['gyro_y']**2 + df['gyro_z']**2)
    
    # FIX: Added 'gyro_mag' to this list so it gets filtered properly!
    columns_to_filter = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z', 'accel_mag', 'gyro_mag']
    
    # Apply low-pass filtering to raw axes to strip out high-frequency noise
    for col in columns_to_filter:
        df[f'{col}_filtered'] = butter_lowpass_filter(df[col].values)
    
    # Rolling window aggregations (captures momentum over recent time steps)
    feature_cols = ['accel_mag_filtered', 'gyro_mag_filtered', 'accel_x_filtered', 'accel_y_filtered']
    for col in feature_cols:
        df[f'{col}_rolling_mean'] = df[col].rolling(window=window_size, min_periods=1).mean()
        df[f'{col}_rolling_std'] = df[col].rolling(window=window_size, min_periods=1).std().fillna(0)
        
    return df