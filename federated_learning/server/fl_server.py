import flwr as fl
import json
import os
import shutil
import glob
import re
from typing import List, Tuple, Dict, Optional
from flwr.common import Metrics
from models.lstm_autoencoder import build_lstm_autoencoder

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, companies, window_size, n_features, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.companies = companies
        self.window_size = window_size
        self.n_features = n_features
        self.current_version = self._get_latest_version()

    def _get_latest_version(self):
        os.makedirs("models/weights", exist_ok=True)
        files = glob.glob("models/weights/Global_Model_v*.weights.h5")
        max_v = 1
        for f in files:
            m = re.search(r"Global_Model_v(\d+)\.weights\.h5", f)
            if m:
                v = int(m.group(1))
                if v > max_v:
                    max_v = v
        return max_v

    def aggregate_fit(self, server_round: int, results, failures):
        # Call the parent aggregate_fit
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_parameters is not None:
            # Convert parameters to weights
            aggregated_ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)
            
            # Save the new global model version
            self.current_version += 1
            save_path = f"models/weights/Global_Model_v{self.current_version}.weights.h5"
            
            model = build_lstm_autoencoder(seq_len=self.window_size, n_features=self.n_features)
            model.set_weights(aggregated_ndarrays)
            model.save_weights(save_path)
            
            print(f"Saved Global Model v{self.current_version} to {save_path}")
            
            # Distribute to participating companies
            for comp in self.companies:
                comp_model_dir = f"companies/{comp}/models"
                os.makedirs(comp_model_dir, exist_ok=True)
                dest_path = os.path.join(comp_model_dir, "local_model.weights.h5")
                shutil.copy2(save_path, dest_path)
                print(f"Distributed Global Model v{self.current_version} to {comp}")

        return aggregated_parameters, aggregated_metrics

def get_server_strategy(num_clients, companies, window_size=24, n_features=2):
    round_metrics = []
    if os.path.exists("eval/fl_round_log.json"):
        try:
            with open("eval/fl_round_log.json", "r") as f:
                round_metrics = json.load(f)
        except Exception:
            pass
    
    def evaluate_metrics_aggregation_fn(results: List[Tuple[int, Metrics]]) -> Metrics:
        if not results:
            return {}
        
        total_examples = sum([num_examples for num_examples, _ in results])
        weighted_f1 = sum([num_examples * m.get("f1", 0.0) for num_examples, m in results]) / total_examples
        weighted_loss = sum([num_examples * m.get("loss", 0.0) for num_examples, m in results]) / total_examples
        
        print(f"Aggregated F1: {weighted_f1:.4f} | Aggregated Loss: {weighted_loss:.6f}")
        
        round_metrics.append({"f1": float(weighted_f1), "loss": float(weighted_loss)})
        os.makedirs("eval", exist_ok=True)
        with open("eval/fl_round_log.json", "w") as f:
            json.dump(round_metrics, f, indent=4)
            
        return {"f1": weighted_f1, "loss": weighted_loss}

    def fit_config(server_round: int):
        return {
            "epochs": 1,
            "batch_size": 32,
        }

    strategy = SaveModelStrategy(
        companies=companies,
        window_size=window_size,
        n_features=n_features,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        on_fit_config_fn=fit_config,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn
    )
    
    return strategy
