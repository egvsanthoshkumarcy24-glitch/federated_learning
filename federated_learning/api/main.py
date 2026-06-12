import os
import json
import shutil
import subprocess
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import pandas as pd
import numpy as np

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.lstm_autoencoder import build_lstm_autoencoder
from shared.dataset_manager import validate_and_preprocess
from tensorflow.keras.callbacks import EarlyStopping

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_config(company: str) -> dict:
    path = f"companies/{company}/config.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

def save_config(company: str, data: dict):
    path = f"companies/{company}/config.json"
    cfg = load_config(company)
    cfg.update(data)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=4)

@app.get("/api/companies")
def get_companies():
    if not os.path.exists("companies"):
        os.makedirs("companies")
    
    comps = sorted([d for d in os.listdir("companies") if os.path.isdir(os.path.join("companies", d))])
    
    if not comps:
        for c in ["Company_A", "Company_B", "Company_C"]:
            os.makedirs(os.path.join("companies", c))
        comps = ["Company_A", "Company_B", "Company_C"]
    
    res = []
    color_map = {'Company_A':'#00d4ff', 'Company_B':'#7c3aed', 'Company_C':'#00ff88', 'Company_D':'#f59e0b', 'Company_E':'#ff4444'}
    for idx, c in enumerate(comps):
        cfg = load_config(c)
        res.append({
            "id": c[-1] if '_' in c else c[0],
            "name": c,
            "color": color_map.get(c, '#00d4ff'),
            "sensors": cfg.get("feature_cols", []),
            "status": "active" if "dataset_path" in cfg else "idle",
            "lastSync": "Just now",
            "config": cfg
        })
    return res

@app.get("/api/config/{company}")
def get_config(company: str):
    return load_config(company)

@app.get("/api/dataset-info/{company}")
def get_dataset_info(company: str):
    cfg = load_config(company)
    if "dataset_path" not in cfg or not os.path.exists(cfg["dataset_path"]):
        raise HTTPException(status_code=400, detail="Dataset not found")
    df = pd.read_csv(cfg["dataset_path"], nrows=5)
    return {"status": "success", "all_columns": df.columns.tolist()}

@app.post("/api/config/{company}")
def update_config(company: str, config: dict):
    save_config(company, config)
    return {"status": "success"}

@app.post("/api/upload/{company}")
async def upload_dataset(company: str, file: UploadFile = File(...)):
    dest_dir = f"companies/{company}/datasets"
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, file.filename)
    
    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    df = pd.read_csv(dest_path)
    cols = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    ts_candidates = [c for c in cols if 'time' in c.lower() or 'date' in c.lower() or 'ts' in c.lower()]
    ts_col = ts_candidates[0] if ts_candidates else None
    
    label_candidates = [c for c in cols if 'label' in c.lower() or 'target' in c.lower() or 'anomaly' in c.lower() or 'class' in c.lower()]
    label_col = label_candidates[0] if label_candidates else None
    
    suggested_features = [c for c in numeric_cols if c != ts_col and c != label_col]
    
    cfg_update = {
        "dataset_path": dest_path
    }
    save_config(company, cfg_update)
    
    return {
        "status": "success", 
        "filename": file.filename, 
        "all_columns": cols,
        "suggested_timestamp": ts_col,
        "suggested_features": suggested_features,
        "suggested_label": label_col
    }

@app.post("/api/train/{company}")
def train_local_model(company: str, params: dict):
    cfg = load_config(company)
    if "dataset_path" not in cfg:
        raise HTTPException(status_code=400, detail="No dataset configured for this company.")
        
    df = pd.read_csv(cfg["dataset_path"])
    window_size = params.get("window_size", 24)
    epochs = params.get("epochs", 10)
    batch_size = params.get("batch_size", 32)
    
    X_windows, _, scaler = validate_and_preprocess(
        df, cfg.get("timestamp_col"), cfg.get("feature_cols", []), window_size,
        save_dir=f"companies/{company}/models", is_train=True
    )
    
    n_features = len(cfg.get("feature_cols", []))
    model = build_lstm_autoencoder(seq_len=window_size, n_features=n_features)
    es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    history = model.fit(
        X_windows, X_windows,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[es],
        verbose=0
    )
    
    model.save_weights(f"companies/{company}/models/local_model.weights.h5")
    save_config(company, {"window_size": window_size, "n_features": n_features})
    
    return {
        "status": "success",
        "history": [
            {"epoch": i+1, "trainLoss": history.history['loss'][i], "valLoss": history.history['val_loss'][i]}
            for i in range(len(history.history['loss']))
        ]
    }

@app.post("/api/detect/{company}")
def detect_anomalies(company: str, params: dict):
    cfg = load_config(company)
    if not os.path.exists(f"companies/{company}/models/local_model.weights.h5"):
        raise HTTPException(status_code=400, detail="Local model not trained.")
        
    df = pd.read_csv(cfg["dataset_path"])
    window_size = cfg.get("window_size", 24)
    threshold = params.get("threshold", 0.05)
    
    X_windows, df_windows, scaler = validate_and_preprocess(
        df, cfg.get("timestamp_col"), cfg.get("feature_cols", []), window_size,
        save_dir=f"companies/{company}/models", is_train=False
    )
    
    model = build_lstm_autoencoder(seq_len=window_size, n_features=cfg.get("n_features", 2))
    model.load_weights(f"companies/{company}/models/local_model.weights.h5")
    
    preds = model.predict(X_windows, verbose=0)
    mse = np.mean(np.power(X_windows - preds, 2), axis=(1, 2))
    
    results = []
    ts_col = cfg.get("timestamp_col")
    
    step = max(1, len(mse) // 200) 
    
    for i in range(0, len(mse), step):
        t_val = str(df_windows.iloc[i][ts_col]) if ts_col else f"Seq_{i}"
        results.append({
            "time": t_val,
            "mse": float(mse[i]),
            "isAnomaly": bool(mse[i] > threshold),
            "threshold": threshold
        })
        
    anom_count = int(np.sum((mse > threshold).astype(int)))
    total_count = len(mse)
        
    return {
        "status": "success",
        "total_anomalies": anom_count,
        "total_scanned": total_count,
        "results": results
    }

@app.post("/api/federate")
def trigger_federation(params: dict):
    comps = sorted([d for d in os.listdir("companies") if os.path.isdir(os.path.join("companies", d))])
    active_comps = [c for c in comps if load_config(c).get("dataset_path")]
    
    if len(active_comps) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 active companies with datasets to run FL.")
        
    cmd = ["python", "run_federated_streamlit.py", "--companies"] + active_comps + \
          ["--window_size", str(params.get("window_size", 24)), 
           "--n_features", str(params.get("n_features", 2)), 
           "--num_rounds", str(params.get("num_rounds", 3))]
           
    subprocess.Popen(cmd)
    return {"status": "started"}

@app.get("/api/federate/status")
def get_federation_status():
    fl_rounds = 0
    fl_log = []
    if os.path.exists("eval/fl_round_log.json"):
        try:
            with open("eval/fl_round_log.json", "r") as f:
                fl_log = json.load(f)
            fl_rounds = len(fl_log)
        except: pass

    max_v = 0
    versions = []
    if os.path.exists("models/weights"):
        for f in os.listdir("models/weights"):
            if f.startswith("Global_Model_v") and f.endswith(".weights.h5"):
                v = int(f.replace("Global_Model_v", "").replace(".weights.h5", ""))
                max_v = max(max_v, v)
                versions.append(v)
                
    versions.sort()
    
    conv_data = []
    for i, round_data in enumerate(fl_log):
        metric_key = "loss" if "loss" in round_data else "f1"
        conv_data.append({
            "round": i+1,
            "loss": round_data.get(metric_key, 0)
        })
        
    model_versions = []
    for v in versions[-11:]:
        model_versions.append({
            "version": f"v{v}",
            "round": v,
            "isCurrent": v == max_v
        })
        
    return {
        "status": "success",
        "rounds_completed": fl_rounds,
        "latest_model_version": max_v,
        "convergence_data": conv_data,
        "model_versions": model_versions
    }
