from model_pipeline import train_local
from client_data import create_clients
import pandas as pd

# Load your dataset here
df = pd.read_csv("your_dataset.csv")

features = ['temperature', 'humidity']

clients = create_clients(df, features)

client_models = {}

for device, data in clients.items():
    print(f"Training {device}")
    model = train_local(data["X_train"][:30000])
    client_models[device] = model