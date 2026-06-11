import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

def create_sequences(data, labels, seq_len=24):
    X = []
    y = []
    # We want sliding windows of seq_len
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len])
        # A sequence is anomalous if any point in the sequence is anomalous
        y.append(1 if np.sum(labels[i : i + seq_len]) > 0 else 0)
    return np.array(X), np.array(y)

def get_dataloader(node_dir, seq_len=24):
    """
    Returns scaled X_train, y_train, X_val, y_val, X_test, y_test for the given node.
    """
    train_df = pd.read_csv(os.path.join(node_dir, 'train.csv'))
    val_df = pd.read_csv(os.path.join(node_dir, 'val.csv'))
    test_df = pd.read_csv(os.path.join(node_dir, 'test.csv'))
    
    features = ['temperature', 'humidity']
    
    scaler = MinMaxScaler()
    # Fit scaler only on train
    train_scaled = scaler.fit_transform(train_df[features])
    val_scaled = scaler.transform(val_df[features])
    test_scaled = scaler.transform(test_df[features])
    
    # Save scaler
    joblib.dump(scaler, os.path.join(node_dir, 'scaler.pkl'))
    
    X_train, y_train = create_sequences(train_scaled, train_df['anomaly'].values, seq_len)
    X_val, y_val = create_sequences(val_scaled, val_df['anomaly'].values, seq_len)
    X_test, y_test = create_sequences(test_scaled, test_df['anomaly'].values, seq_len)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = get_dataloader("data/node_1")
    print(f"Node 1 shapes: X_train={X_train.shape}, y_train={y_train.shape}")
