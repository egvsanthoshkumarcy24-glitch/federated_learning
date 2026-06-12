import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

def create_sequences(data, window_size):
    """
    Creates overlapping sliding windows of data.
    """
    X = []
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size])
    return np.array(X)

def validate_and_preprocess(df, timestamp_col, feature_cols, window_size, save_dir=None, is_train=True, scaler=None):
    """
    Validates the dataframe, scales the features, and generates sliding windows.
    """
    # Validation
    if timestamp_col and timestamp_col != "<None (Use Row Order)>":
        if timestamp_col not in df.columns:
            raise ValueError(f"Timestamp column '{timestamp_col}' not found.")
    for col in feature_cols:
        if col not in df.columns:
            raise ValueError(f"Feature column '{col}' not found.")
            
    df = df.copy()
    
    # Sort by timestamp if provided
    if timestamp_col and timestamp_col != "<None (Use Row Order)>":
        try:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            df = df.sort_values(by=timestamp_col).reset_index(drop=True)
        except Exception as e:
            raise ValueError(f"Could not parse timestamp column: {e}")
        
    # Ensure features are numeric
    try:
        df[feature_cols] = df[feature_cols].apply(pd.to_numeric)
    except Exception as e:
        raise ValueError(f"One or more feature columns contain non-numeric data that cannot be converted to numbers: {e}")
        
    # Check for missing values in features
    if df[feature_cols].isnull().any().any():
        df[feature_cols] = df[feature_cols].interpolate(method='linear').bfill().ffill()

    # Scaling
    if is_train:
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[feature_cols])
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            joblib.dump(scaler, os.path.join(save_dir, 'scaler.pkl'))
    else:
        if scaler is None:
            if save_dir and os.path.exists(os.path.join(save_dir, 'scaler.pkl')):
                scaler = joblib.load(os.path.join(save_dir, 'scaler.pkl'))
            else:
                raise ValueError("Scaler must be provided for inference.")
        scaled_data = scaler.transform(df[feature_cols])

    # Window Generation
    X_windows = create_sequences(scaled_data, window_size)
    
    # We also return the timestamps and raw data corresponding to the end of the windows for reporting
    valid_indices = range(window_size, len(df))
    df_windows = df.iloc[valid_indices].reset_index(drop=True)
    
    return X_windows, df_windows, scaler
