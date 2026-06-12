import os
import json
import flwr as fl
from clients.flower_client import FlowerClient
from server.fl_server import get_server_strategy
from models.lstm_autoencoder import build_lstm_autoencoder
from data.dataloader import get_dataloader

def start_simulation(num_rounds=30):
    os.makedirs("eval", exist_ok=True)
    
    # Load data for all clients
    client_data = {}
    for i in range(1, 6):
        node_dir = f"data/node_{i}"
        if os.path.exists(node_dir):
            X_train, y_train, X_val, y_val, X_test, y_test = get_dataloader(node_dir)
            client_data[i] = (X_train, X_val, y_val)
    
    def client_fn(cid: str) -> fl.client.Client:
        node_id = int(cid) + 1 # cid is 0-4
        X_train, X_val, y_val = client_data[node_id]
        model = build_lstm_autoencoder(seq_len=24, n_features=2)
        return FlowerClient(model, X_train, X_val, y_val).to_client()
        
    strategy = get_server_strategy()
    
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(client_data),
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1}
    )
    
    print("Simulation complete. Logs saved to eval/fl_round_log.json")

    # Final Evaluation of the Federated Model on all Test Sets
    evaluate_federated_model()

def evaluate_federated_model():
    # To evaluate the global model, we just run one round of get_weights conceptually, 
    # but in simulation it's easier to just save weights or if we ran an actual server.
    # For now, we will assume we can extract the best F1 from fl_round_log.
    pass

if __name__ == "__main__":
    start_simulation(num_rounds=30)
