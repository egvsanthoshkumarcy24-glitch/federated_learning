import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping
import sys

# Ensure imports work when running streamlit from root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.lstm_autoencoder import build_lstm_autoencoder
from shared.dataset_manager import validate_and_preprocess
from detection_utils import detect_anomalies
from shared.report_generator import generate_pdf_report

st.set_page_config(layout="wide", page_title="Federated FL Platform")

# ----------------- AESTHETICS & CSS -----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}
.main {
    background-color: #0d1117;
    color: #c9d1d9;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    white-space: pre-wrap;
    background-color: #161b22;
    border-radius: 4px 4px 0px 0px;
    color: #8b949e;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background-color: #21262d;
    color: #58a6ff;
    border-top: 2px solid #58a6ff;
}
.glass-container {
    background: rgba(22, 27, 34, 0.7);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 24px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    transition: transform 0.2s ease-in-out;
}
.glass-container:hover {
    transform: translateY(-2px);
}
.metric-value {
    font-size: 36px;
    font-weight: 800;
    color: #58a6ff;
}
.metric-label {
    font-size: 14px;
    font-weight: 600;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 1px;
}
h1, h2, h3 {
    color: #c9d1d9 !important;
}
hr {
    border-color: #30363d;
}
</style>
""", unsafe_allow_html=True)

if not os.path.exists("companies"):
    os.makedirs("companies")
    
COMPANIES = sorted([d for d in os.listdir("companies") if os.path.isdir(os.path.join("companies", d))])

for c in COMPANIES:
    os.makedirs(f"companies/{c}/datasets", exist_ok=True)
    os.makedirs(f"companies/{c}/models", exist_ok=True)
    os.makedirs(f"companies/{c}/reports", exist_ok=True)

def load_config(company):
    path = f"companies/{company}/config.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

def save_config(company, data):
    path = f"companies/{company}/config.json"
    cfg = load_config(company)
    cfg.update(data)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=4)

# ----------------- SIDEBAR NAVIGATION -----------------
st.sidebar.title("Navigation")
nav_choice = st.sidebar.radio("Go to", ["Admin Dashboard"] + COMPANIES)

# ======================================================================
#                            ADMIN DASHBOARD
# ======================================================================
if nav_choice == "Admin Dashboard":
    st.markdown("<h1><span style='color:#58a6ff'>Admin</span> Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8b949e'>Federated Learning Orchestration and Global Metrics</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    # Calculate active companies
    def is_company_active(comp):
        cfg = load_config(comp)
        has_dataset = bool(cfg.get("dataset_path"))
        has_model = os.path.exists(f"companies/{comp}/models/local_model.weights.h5")
        return has_dataset or has_model
    
    active = sum([1 for c in COMPANIES if is_company_active(c)])
    with col1:
        st.markdown(f"<div class='glass-container'><div class='metric-label'>Active Companies</div><div class='metric-value'>{active} / {len(COMPANIES)}</div></div>", unsafe_allow_html=True)
        
    # Determine latest global model version
    max_v = 0
    if os.path.exists("models/weights"):
        for f in os.listdir("models/weights"):
            if f.startswith("Global_Model_v") and f.endswith(".weights.h5"):
                v = int(f.replace("Global_Model_v", "").replace(".weights.h5", ""))
                max_v = max(max_v, v)
                
    with col2:
        val_str = f"v{max_v}" if max_v > 0 else "Missing"
        st.markdown(f"<div class='glass-container'><div class='metric-label'>Latest Global Model</div><div class='metric-value'>{val_str}</div></div>", unsafe_allow_html=True)
        
    fl_rounds = 0
    if os.path.exists("eval/fl_round_log.json"):
        try:
            with open("eval/fl_round_log.json", "r") as f:
                fl_log = json.load(f)
            fl_rounds = len(fl_log)
        except: pass
        
    with col3:
        st.markdown(f"<div class='glass-container'><div class='metric-label'>FL Rounds Completed</div><div class='metric-value'>{fl_rounds}</div></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Federated Learning Orchestration")
    
    fl_col1, fl_col2 = st.columns([1, 2])
    with fl_col1:
        rounds_input = st.number_input("Number of FL Rounds", min_value=1, max_value=50, value=3)
        window_size_input = st.number_input("Window Size (must match clients)", min_value=1, value=24)
        n_features_input = st.number_input("Number of Features (must match clients)", min_value=1, value=2)
        
        if st.button("Trigger Federated Round", type="primary"):
            if active < 2:
                st.error("Need at least 2 active companies with uploaded datasets to run FL.")
            else:
                st.info("Starting Federated Learning simulation in background...")
                # We trigger the run_federated_streamlit.py script
                active_comps = [c for c in COMPANIES if load_config(c).get("dataset_path")]
                cmd = ["python", "run_federated_streamlit.py", "--companies"] + active_comps + \
                      ["--window_size", str(window_size_input), "--n_features", str(n_features_input), "--num_rounds", str(rounds_input)]
                with st.spinner("Running Federated Learning simulation..."):
                    subprocess.run(cmd)
                st.success("Federated Learning process completed.")
                st.rerun()

    with fl_col2:
        if fl_rounds > 0:
            metric_key = "loss" if "loss" in fl_log[0] else "f1"
            metric_label = "Reconstruction Loss (MSE)" if metric_key == "loss" else "F1 Score"
            st.subheader(f"Global Convergence ({metric_label})")
            rounds = list(range(1, fl_rounds + 1))
            vals = [x.get(metric_key, 0) for x in fl_log]
            fig, ax = plt.subplots(figsize=(10, 4))
            fig.patch.set_facecolor('#0d1117')
            ax.set_facecolor('#0d1117')
            ax.plot(rounds, vals, marker='o', color='#58a6ff', linewidth=2)
            ax.set_xlabel("Round", color='#c9d1d9')
            ax.set_ylabel(metric_label, color='#c9d1d9')
            ax.tick_params(colors='#c9d1d9')
            for spine in ax.spines.values():
                spine.set_edgecolor('#30363d')
            st.pyplot(fig)
        else:
            st.info("No FL rounds logged yet.")

# ======================================================================
#                              COMPANY VIEW
# ======================================================================
else:
    company = nav_choice
    st.markdown(f"<h1><span style='color:#58a6ff'>{company}</span> Workspace</h1>", unsafe_allow_html=True)
    
    cfg = load_config(company)
    
    tab_upload, tab_train, tab_detect, tab_reports = st.tabs([
        "📁 Upload Dataset", "🧠 Train Local Model", "🔍 Detect Anomalies", "📄 Reports"
    ])
    
    # --- TAB 1: UPLOAD DATASET ---
    with tab_upload:
        st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
        st.subheader("Data Discovery & Configuration")
        
        # In a real app, file_uploader would be used. Since this is folder-based simulation:
        dataset_options = []
        for root, dirs, files in os.walk("datasets"):
            for f in files:
                if f.endswith('.csv'):
                    dataset_options.append(os.path.join(root, f))
                    
        selected_file = st.selectbox("Select Dataset from discovered sources:", dataset_options, 
                                     index=dataset_options.index(cfg["dataset_path"]) if "dataset_path" in cfg and cfg["dataset_path"] in dataset_options else 0)
        
        if selected_file:
            df = pd.read_csv(selected_file)
            st.write("Preview:")
            st.dataframe(df.head())
            
            cols = df.columns.tolist()
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            ts_candidates = [c for c in cols if 'time' in c.lower() or 'date' in c.lower() or 'ts' in c.lower()]
            ts_options = ["<None (Use Row Order)>"] + cols
            
            if "timestamp_col" in cfg and cfg["timestamp_col"] in ts_options:
                default_ts_index = ts_options.index(cfg["timestamp_col"])
            else:
                default_ts_index = ts_options.index(ts_candidates[0]) if ts_candidates else 0
                
            if default_ts_index == 0:
                st.info("No timestamp column detected. Using dataset row order.")
                
            ts_col = st.selectbox("Timestamp Column (used for ordering only)", ts_options, index=default_ts_index)
            feat_cols = st.multiselect("Feature Columns (for model training)", numeric_cols, 
                                       default=[c for c in cfg.get("feature_cols", []) if c in numeric_cols])
            
            if st.button("Save Configuration", type="primary"):
                save_config(company, {"dataset_path": selected_file, "timestamp_col": ts_col, "feature_cols": feat_cols})
                st.success("Configuration saved.")
        st.markdown("</div>", unsafe_allow_html=True)

    # --- TAB 2: TRAIN LOCAL MODEL ---
    with tab_train:
        st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
        if "dataset_path" not in cfg or not cfg["feature_cols"]:
            st.warning("Please configure your dataset in the Upload tab first.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                epochs = st.number_input("Epochs", min_value=1, value=10)
                batch_size = st.number_input("Batch Size", min_value=8, value=32)
            with col2:
                window_size = st.number_input("Window Size", min_value=1, value=24)
                learning_rate = st.selectbox("Learning Rate", [0.01, 0.001, 0.0001], index=1)
                
            if st.button("Train Local Model", type="primary"):
                with st.spinner("Preprocessing & Training..."):
                    df = pd.read_csv(cfg["dataset_path"])
                    # Use shared dataset manager
                    X_windows, _, scaler = validate_and_preprocess(
                        df, cfg["timestamp_col"], cfg["feature_cols"], window_size, 
                        save_dir=f"companies/{company}/models", is_train=True
                    )
                    
                    n_features = len(cfg["feature_cols"])
                    model = build_lstm_autoencoder(seq_len=window_size, n_features=n_features)
                    
                    es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
                    # For demonstration, use a subset if too large or just train
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
                    
                    st.success("Training complete!")
                    
                    fig, ax = plt.subplots(figsize=(8, 3))
                    fig.patch.set_facecolor('#0d1117')
                    ax.set_facecolor('#0d1117')
                    ax.plot(history.history['loss'], label='Train Loss', color='#58a6ff')
                    ax.plot(history.history['val_loss'], label='Val Loss', color='#3fb950')
                    ax.legend()
                    for spine in ax.spines.values(): spine.set_edgecolor('#30363d')
                    ax.tick_params(colors='#c9d1d9')
                    st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    # --- TAB 3: DETECT ANOMALIES ---
    with tab_detect:
        st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
        if not os.path.exists(f"companies/{company}/models/local_model.weights.h5"):
            st.warning("Please train the local model first.")
        else:
            threshold = st.slider("Anomaly Threshold", 0.001, 0.1, 0.03, step=0.005)
            
            if st.button("Run Anomaly Detection", type="primary"):
                with st.spinner("Detecting anomalies..."):
                    df = pd.read_csv(cfg["dataset_path"])
                    window_size = cfg.get("window_size", 24)
                    X_windows, df_windows, scaler = validate_and_preprocess(
                        df, cfg["timestamp_col"], cfg["feature_cols"], window_size, 
                        save_dir=f"companies/{company}/models", is_train=False
                    )
                    
                    model = build_lstm_autoencoder(seq_len=window_size, n_features=cfg["n_features"])
                    model.load_weights(f"companies/{company}/models/local_model.weights.h5")
                    
                    preds = model.predict(X_windows, verbose=0)
                    mse = np.mean(np.power(X_windows - preds, 2), axis=(1, 2))
                    
                    anomalies = (mse > threshold).astype(int)
                    df_windows['mse'] = mse
                    df_windows['anomaly_pred'] = anomalies
                    
                    anomaly_count = int(np.sum(anomalies))
                    anomaly_pct = (anomaly_count / len(anomalies)) * 100
                    
                    # Update config for report
                    save_config(company, {"anomaly_count": anomaly_count, "anomaly_pct": anomaly_pct})
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Anomalies", anomaly_count)
                    with col2:
                        st.metric("Anomaly Percentage", f"{anomaly_pct:.2f}%")
                        
                    fig, ax = plt.subplots(figsize=(12, 4))
                    fig.patch.set_facecolor('#0d1117')
                    ax.set_facecolor('#0d1117')
                    
                    ax.plot(df_windows[cfg["timestamp_col"]], df_windows['mse'], color='#8b949e', alpha=0.6, label='Reconstruction MSE')
                    ax.axhline(threshold, color='#f85149', linestyle='--', label='Threshold')
                    
                    anom_df = df_windows[df_windows['anomaly_pred'] == 1]
                    if not anom_df.empty:
                        ax.scatter(anom_df[cfg["timestamp_col"]], anom_df['mse'], color='#f85149', label='Anomalies')
                        
                    ax.set_xlabel("Time", color='#c9d1d9')
                    ax.set_ylabel("MSE", color='#c9d1d9')
                    ax.tick_params(colors='#c9d1d9')
                    for spine in ax.spines.values(): spine.set_edgecolor('#30363d')
                    ax.legend()
                    st.pyplot(fig)
                    
                    # Store results temporarily to CSV for reporting
                    df_windows.to_csv(f"companies/{company}/reports/latest_detection.csv", index=False)
                    st.success("Anomaly detection completed. Results saved.")
        st.markdown("</div>", unsafe_allow_html=True)

    # --- TAB 4: REPORTS ---
    with tab_reports:
        st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
        st.subheader("Generate PDF Report")
        if st.button("Generate Report", type="primary"):
            if "anomaly_count" not in cfg:
                st.warning("Please run anomaly detection first.")
            else:
                with st.spinner("Generating PDF..."):
                    summary = {
                        "Dataset Path": cfg.get("dataset_path", "N/A"),
                        "Window Size": cfg.get("window_size", "N/A")
                    }
                    pdf_path = generate_pdf_report(
                        company_name=company,
                        dataset_summary=summary,
                        features=cfg.get("feature_cols", []),
                        anomalies_count=cfg.get("anomaly_count", 0),
                        anomalies_percentage=cfg.get("anomaly_pct", 0.0),
                        output_dir=f"companies/{company}/reports"
                    )
                    st.success(f"Report generated: {pdf_path}")
                    with open(pdf_path, "rb") as f:
                        st.download_button("Download PDF", f, file_name=os.path.basename(pdf_path))
        st.markdown("</div>", unsafe_allow_html=True)
