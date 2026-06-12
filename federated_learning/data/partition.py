import os
import pandas as pd
from sklearn.model_selection import train_test_split

def partition_data(input_path="data/processed/clean_data.csv", output_dir="data"):
    df = pd.read_csv(input_path)
    
    facilities = df['facility_type'].unique()
    
    for i, facility in enumerate(facilities):
        node_id = i + 1
        node_dir = os.path.join(output_dir, f"node_{node_id}")
        os.makedirs(node_dir, exist_ok=True)
        
        node_df = df[df['facility_type'] == facility].copy()
        
        # Sort by time to preserve sequence
        node_df = node_df.sort_values('timestamp')
        
        # Split: 70% train, 15% val, 15% test
        n_total = len(node_df)
        n_train = int(n_total * 0.7)
        n_val = int(n_total * 0.15)
        
        train_df = node_df.iloc[:n_train]
        val_df = node_df.iloc[n_train:n_train+n_val]
        test_df = node_df.iloc[n_train+n_val:]
        
        train_df.to_csv(os.path.join(node_dir, "train.csv"), index=False)
        val_df.to_csv(os.path.join(node_dir, "val.csv"), index=False)
        test_df.to_csv(os.path.join(node_dir, "test.csv"), index=False)
        
        print(f"Node {node_id} ({facility}): Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

if __name__ == "__main__":
    partition_data()
