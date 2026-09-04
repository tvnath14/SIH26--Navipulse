import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Import our custom preprocessing module from utils.py
from utils import extract_rolling_features

def train_production_idr_model():
    print("=== Step 1: Initializing IO-VNBD AI Pipeline ===")
    
    data_filename = "C:\Users\Vishwas BVM\Desktop\SIH\S-M.txt"
    
    try:
        df = pd.read_csv(data_filename)
        print(f"Successfully loaded '{data_filename}' with shape: {df.shape}")
        
        required_cols = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z', 'speed']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing expected dataset column: {col}")
                
    except (FileNotFoundError, ValueError) as e:
        print(f"⚠️ Notice: {e}")
        print("Generating synthetic IO-VNBD schema structure for local pipeline testing...")
        
        np.random.seed(42)
        n_samples = 1000
        df = pd.DataFrame({
            'accel_x': np.random.normal(0, 0.5, n_samples),
            'accel_y': np.random.normal(0, 0.5, n_samples),
            'accel_z': np.random.normal(9.81, 0.2, n_samples),
            'gyro_x': np.random.normal(0, 0.05, n_samples),
            'gyro_y': np.random.normal(0, 0.05, n_samples),
            'gyro_z': np.random.normal(0, 0.05, n_samples),
            'speed': np.abs(np.random.normal(20, 5, n_samples))
        })

    # Preprocess and extract features using the function from utils.py
    processed_df = extract_rolling_features(df, window_size=5)
    
    feature_columns = [col for col in processed_df.columns if col not in ['speed', 'timestamp', 'lat', 'lon']]
    X = processed_df[feature_columns]
    y = processed_df['speed']

    print(f"Total features engineered for training: {len(feature_columns)}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("=== Step 2: Training Optimized Random Forest Regressor ===")
    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f"Training Complete! Test Root Mean Squared Error (RMSE): {rmse:.4f} km/h")

    model_filename = 'speed_model.pkl'
    with open(model_filename, 'wb') as file:
        pickle.dump({'model': model, 'features': feature_columns}, file)
    
    print(f"=== Step 3: Artifact Saved Successfully as '{model_filename}' ===")

if __name__ == "__main__":
    train_production_idr_model()