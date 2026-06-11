import os
import json
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from models.lstm_autoencoder import build_lstm_autoencoder

# Since this is run from federated_learning root sometimes, need to handle paths
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.dataloader import get_dataloader

def train(model, X_train, X_val, epochs=10, batch_size=32):
    es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=1
    )
    return model, history

def compute_threshold(model, X_train):
    preds = model.predict(X_train)
    mse = np.mean(np.power(X_train - preds, 2), axis=(1, 2))
    threshold = np.percentile(mse, 99)
    return threshold

def evaluate_model(model, X_test, y_test, threshold):
    from sklearn.metrics import f1_score, precision_score, recall_score
    preds = model.predict(X_test)
    mse = np.mean(np.power(X_test - preds, 2), axis=(1, 2))
    
    y_pred = (mse > threshold).astype(int)
    
    f1 = f1_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    
    return f1, precision, recall

def run_isolated_baseline():
    os.makedirs("eval", exist_ok=True)
    results = {}
    
    for i in range(1, 6):
        node_dir = f"data/node_{i}"
        if not os.path.exists(node_dir):
            continue
            
        print(f"--- Training Isolated Baseline for Node {i} ---")
        X_train, y_train, X_val, y_val, X_test, y_test = get_dataloader(node_dir)
        
        model = build_lstm_autoencoder(seq_len=24, n_features=2)
        model, _ = train(model, X_train, X_val, epochs=10)
        
        # Save weights
        os.makedirs("models/weights", exist_ok=True)
        model.save_weights(f"models/weights/node_{i}.weights.h5")
        
        threshold = compute_threshold(model, X_train)
        f1, precision, recall = evaluate_model(model, X_test, y_test, threshold)
        
        results[f"node_{i}"] = {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "threshold": threshold
        }
        print(f"Node {i} Results: F1={f1:.4f}, Prec={precision:.4f}, Rec={recall:.4f}")
        
    with open("eval/isolated_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
if __name__ == "__main__":
    run_isolated_baseline()
