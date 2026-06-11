import flwr as fl
import json
import os
from typing import List, Tuple, Dict, Optional
from flwr.common import Metrics

def get_server_strategy():
    # Save round metrics to file
    round_metrics = []
    
    def evaluate_metrics_aggregation_fn(results: List[Tuple[int, Metrics]]) -> Metrics:
        if not results:
            return {}
        
        total_examples = sum([num_examples for num_examples, _ in results])
        weighted_f1 = sum([num_examples * m["f1"] for num_examples, m in results]) / total_examples
        
        print(f"Aggregated F1: {weighted_f1:.4f}")
        
        round_metrics.append({"f1": float(weighted_f1)})
        os.makedirs("eval", exist_ok=True)
        with open("eval/fl_round_log.json", "w") as f:
            json.dump(round_metrics, f, indent=4)
            
        return {"f1": weighted_f1}

    def fit_config(server_round: int):
        return {
            "epochs": 1,
            "batch_size": 32,
        }

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=5,
        min_evaluate_clients=5,
        min_available_clients=5,
        on_fit_config_fn=fit_config,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn
    )
    
    return strategy
