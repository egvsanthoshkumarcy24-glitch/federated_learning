import os
import json
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.dataloader import get_dataloader
from models.lstm_autoencoder import build_lstm_autoencoder
from models.train import train, compute_threshold, evaluate_model

def run_centralized_baseline():
    os.makedirs("eval", exist_ok=True)
    
    all_X_train = []
    all_X_val = []
    
    # We evaluate on each node's test set to have comparable metrics to Federated
    node_test_data = {}
    
    for i in range(1, 6):
        node_dir = f"data/node_{i}"
        if not os.path.exists(node_dir):
            continue
            
        X_train, y_train, X_val, y_val, X_test, y_test = get_dataloader(node_dir)
        all_X_train.append(X_train)
        all_X_val.append(X_val)
        
        node_test_data[i] = (X_test, y_test)
        
    if not all_X_train:
        print("No data found.")
        return
        
    X_train_central = np.concatenate(all_X_train, axis=0)
    X_val_central = np.concatenate(all_X_val, axis=0)
    
    print(f"Centralized Training Data: {X_train_central.shape}")
    
    model = build_lstm_autoencoder(seq_len=24, n_features=2)
    model, _ = train(model, X_train_central, X_val_central, epochs=10)
    
    threshold = compute_threshold(model, X_train_central)
    
    results = {}
    for i, (X_test, y_test) in node_test_data.items():
        f1, precision, recall = evaluate_model(model, X_test, y_test, threshold)
        results[f"node_{i}"] = {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "threshold": threshold
        }
        print(f"Node {i} Centralized Results: F1={f1:.4f}, Prec={precision:.4f}, Rec={recall:.4f}")
        
    with open("eval/centralized_results.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    run_centralized_baseline()
