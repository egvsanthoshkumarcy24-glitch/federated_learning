import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_comparison_metrics():
    # Load all 3 json files
    try:
        with open("eval/isolated_results.json", "r") as f:
            isolated = json.load(f)
    except:
        isolated = {}
        
    try:
        with open("eval/centralized_results.json", "r") as f:
            centralized = json.load(f)
    except:
        centralized = {}
        
    # We will assume federated results per node are same structure, 
    # but since run_federated.py just saves fl_round_log, we might just plot the curve
    try:
        with open("eval/fl_round_log.json", "r") as f:
            fl_log = json.load(f)
    except:
        fl_log = []
        
    # Plot Privacy-Utility Tradeoff Curve
    if fl_log and centralized:
        cent_f1 = np.mean([v["f1"] for v in centralized.values()])
        fl_f1 = [round["f1"] for round in fl_log]
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(fl_f1)+1), fl_f1, label="Federated FedAvg", marker='o')
        plt.axhline(cent_f1, color='r', linestyle='--', label="Centralized Baseline (Upper Bound)")
        
        if isolated:
            iso_f1 = np.mean([v["f1"] for v in isolated.values()])
            plt.axhline(iso_f1, color='g', linestyle='-.', label="Isolated Baseline (Lower Bound)")
            
        plt.title("Privacy-Utility Tradeoff (F1 vs Rounds)")
        plt.xlabel("Communication Rounds")
        plt.ylabel("Global F1 Score")
        plt.legend()
        plt.grid(True)
        
        os.makedirs("paper/figures", exist_ok=True)
        plt.savefig("paper/figures/privacy_utility_tradeoff.png")
        plt.close()
        print("Generated paper/figures/privacy_utility_tradeoff.png")

if __name__ == "__main__":
    import numpy as np
    generate_comparison_metrics()
