import os
import pandas as pd
import numpy as np

def inject_anomalies(node_dir="data/node_{id}", num_nodes=5):
    np.random.seed(42)
    
    for i in range(1, num_nodes + 1):
        d = node_dir.format(id=i)
        
        for split in ['train.csv', 'val.csv', 'test.csv']:
            path = os.path.join(d, split)
            if not os.path.exists(path):
                continue
                
            df = pd.read_csv(path)
            
            # Add anomaly column initialized to 0
            df['anomaly'] = 0
            
            # Inject anomalies only into test set (for real evaluation)
            # but maybe a few in val. We generally don't want them in train for an autoencoder
            # Wait, the blueprint says: "random 3% of samples get temperature spike. Verify injection rate ~2-4%".
            # Autoencoders should be trained on mostly normal data. If we inject in train, we assume unsupervised with low contamination.
            # Let's inject across all, but we will use the clean training data for the autoencoder if possible, 
            # or rely on the low 3% contamination.
            
            num_samples = len(df)
            num_anomalies = int(num_samples * 0.03)
            
            # Select random start indices for anomaly events (1-6 hours duration)
            if num_anomalies > 0:
                std_temp = df['temperature'].std()
                if std_temp == 0 or np.isnan(std_temp):
                    std_temp = 1.0
                
                # We inject events, not just single points
                events_injected = 0
                while events_injected < num_anomalies:
                    idx = np.random.randint(0, num_samples - 6)
                    duration = np.random.randint(1, 7) # 1 to 6 hours
                    
                    # +2.5 to +4 sigma
                    spike_magnitude = np.random.uniform(2.5, 4.0) * std_temp
                    
                    df.loc[idx:idx+duration-1, 'temperature'] += spike_magnitude
                    df.loc[idx:idx+duration-1, 'anomaly'] = 1
                    
                    events_injected += duration
            
            df.to_csv(path, index=False)
            print(f"Node {i} {split}: Injected {df['anomaly'].sum()} anomalous points ({df['anomaly'].mean()*100:.2f}%)")

if __name__ == "__main__":
    inject_anomalies("data/node_{id}", 5)
