import os
import argparse
import numpy as np
from models.lstm_autoencoder import build_lstm_autoencoder

def init_global_model(window_size=24, n_features=2, version="v1"):
    """
    Initializes a global model with the specified window_size and n_features.
    Saves the initial weights to models/weights/Global_Model_{version}.weights.h5
    """
    os.makedirs("models/weights", exist_ok=True)
    
    # Build skeleton model
    model = build_lstm_autoencoder(seq_len=window_size, n_features=n_features)
    
    # Optionally, we could train it on some base dataset, but for a generalized
    # platform where features are dynamic, initializing with random weights is
    # the standard starting point before Federated Learning rounds begin.
    
    save_path = f"models/weights/Global_Model_{version}.weights.h5"
    model.save_weights(save_path)
    print(f"Successfully initialized {save_path} with window_size={window_size}, n_features={n_features}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--window_size", type=int, default=24)
    parser.add_argument("--n_features", type=int, default=2)
    parser.add_argument("--version", type=str, default="v1")
    args = parser.parse_args()
    
    init_global_model(args.window_size, args.n_features, args.version)
