import streamlit as st
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide", page_title="Cold Chain Anomaly Detection")

st.title("Federated Learning for Cold Chain Anomaly Detection")

tab1, tab2, tab3 = st.tabs(["Live Anomaly Detection", "Federated Progress", "3-Way Benchmark"])

with tab1:
    st.header("Node View")
    node_id = st.selectbox("Select Node", [1, 2, 3, 4, 5])
    
    data_path = f"data/node_{node_id}/test.csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        st.write(f"Displaying data for Node {node_id}")
        
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df['temperature'], label='Temperature')
        
        anomalies = df[df['anomaly'] == 1]
        ax.scatter(anomalies.index, anomalies['temperature'], color='red', label='True Anomaly')
        
        ax.legend()
        st.pyplot(fig)
    else:
        st.error("Data not found. Run the data pipeline first.")

with tab2:
    st.header("Federated Learning Rounds")
    log_path = "eval/fl_round_log.json"
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            fl_log = json.load(f)
        
        rounds = [i+1 for i in range(len(fl_log))]
        f1_scores = [r["f1"] for r in fl_log]
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(rounds, f1_scores, marker='o')
        ax.set_xlabel("Round")
        ax.set_ylabel("F1 Score")
        ax.set_title("Global Model Convergence")
        st.pyplot(fig)
    else:
        st.warning("Federated learning log not found.")

with tab3:
    st.header("3-Way Benchmark Comparison")
    
    metrics = []
    
    if os.path.exists("eval/isolated_results.json"):
        with open("eval/isolated_results.json", "r") as f:
            iso = json.load(f)
            iso_f1 = sum(v["f1"] for v in iso.values()) / len(iso)
            metrics.append({"Condition": "Isolated", "F1 Score": iso_f1})
            
    if os.path.exists("eval/fl_round_log.json"):
        with open("eval/fl_round_log.json", "r") as f:
            fl = json.load(f)
            metrics.append({"Condition": "Federated (Latest Round)", "F1 Score": fl[-1]["f1"]})
            
    if os.path.exists("eval/centralized_results.json"):
        with open("eval/centralized_results.json", "r") as f:
            cent = json.load(f)
            cent_f1 = sum(v["f1"] for v in cent.values()) / len(cent)
            metrics.append({"Condition": "Centralized", "F1 Score": cent_f1})
            
    if metrics:
        df_metrics = pd.DataFrame(metrics)
        st.bar_chart(df_metrics.set_index("Condition"))
    else:
        st.warning("Benchmark results not found.")
