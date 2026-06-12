import os
import argparse
import flwr as fl
import numpy as np
import pandas as pd
import json

from clients.flower_client import FlowerClient
from server.fl_server import get_server_strategy
from models.lstm_autoencoder import build_lstm_autoencoder
from shared.dataset_manager import validate_and_preprocess

def start_simulation(companies, window_size=24, n_features=2, num_rounds=3):
    os.makedirs("eval", exist_ok=True)
    
    # We will load data dynamically for each participating company
    client_data = {}
    
    for i, company in enumerate(companies):
        company_dir = f"companies/{company}"
        # We need to find the csv and settings for this company.
        # Assume streamlit saved a config.json in companies/Company_X/config.json
        config_path = os.path.join(company_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                cfg = json.load(f)
            
            csv_path = cfg.get("dataset_path")
            timestamp_col = cfg.get("timestamp_col")
            feature_cols = cfg.get("feature_cols")
            
            if csv_path and os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                try:
                    X_windows, df_windows, scaler = validate_and_preprocess(
                        df, timestamp_col, feature_cols, window_size, 
                        save_dir=os.path.join(company_dir, "models"), is_train=True
                    )
                    # We use a dummy y_val of zeros for the client skeleton, 
                    # actual anomaly detection is done separately.
                    y_val = np.zeros(len(X_windows)) 
                    client_data[company] = (X_windows, X_windows, y_val)
                    print(f"Loaded data for {company}: {X_windows.shape}")
                except Exception as e:
                    print(f"Error loading data for {company}: {e}")
                    
    if not client_data:
        print("No companies have valid configurations for Federated Learning.")
        return

    company_names = list(client_data.keys())

    def client_fn(cid: str) -> fl.client.Client:
        company = company_names[int(cid)]
        X_train, X_val, y_val = client_data[company]
        model = build_lstm_autoencoder(seq_len=window_size, n_features=n_features)
        
        # Initialize with the latest global model if it exists
        import glob
        import re
        files = glob.glob("models/weights/Global_Model_v*.weights.h5")
        max_v = 0
        latest_file = None
        for f in files:
            m = re.search(r"Global_Model_v(\d+)\.weights\.h5", f)
            if m:
                v = int(m.group(1))
                if v > max_v:
                    max_v = v
                    latest_file = f
                    
        if latest_file and os.path.exists(latest_file):
            model.load_weights(latest_file)
        
        return FlowerClient(model, X_train, X_val, y_val).to_client()
        
    strategy = get_server_strategy(num_clients=len(company_names), companies=company_names, window_size=window_size, n_features=n_features)
    
    # Flower's start_simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(company_names),
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1}
    )
    
    print("Federated Learning simulation complete.")
    
    # Save a marker for streamlit to know it's done
    with open("eval/fl_status.json", "w") as f:
        json.dump({"status": "completed", "rounds": num_rounds}, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--companies", nargs="+", required=True)
    parser.add_argument("--window_size", type=int, default=24)
    parser.add_argument("--n_features", type=int, default=2)
    parser.add_argument("--num_rounds", type=int, default=3)
    args = parser.parse_args()
    
    start_simulation(args.companies, args.window_size, args.n_features, args.num_rounds)
