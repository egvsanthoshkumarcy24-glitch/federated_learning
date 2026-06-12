import os
import pandas as pd
import numpy as np

def generate_raw_data(output_path="data/raw/synthetic_cold_storage.csv", num_rows=50000):
    np.random.seed(42)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 5 different facility types simulating the 5 nodes
    facility_types = [
        "Dairy Warehouse",
        "Meat Cold Store",
        "Frozen Goods Truck Fleet",
        "Pharmaceutical Cold Chain",
        "Mixed Produce"
    ]
    
    # Base parameters for each facility
    # (base_temp, temp_std, base_humidity, hum_std)
    params = {
        "Dairy Warehouse": (4.0, 0.5, 85.0, 5.0),
        "Meat Cold Store": (-20.0, 3.0, 60.0, 10.0),
        "Frozen Goods Truck Fleet": (-18.0, 5.0, 70.0, 8.0),
        "Pharmaceutical Cold Chain": (2.0, 0.1, 50.0, 2.0),
        "Mixed Produce": (10.0, 2.0, 90.0, 4.0)
    }
    
    records = []
    # Generate continuous timeline
    timestamps = pd.date_range(start="2024-01-01", periods=num_rows // len(facility_types), freq="H")
    
    for facility in facility_types:
        b_temp, s_temp, b_hum, s_hum = params[facility]
        
        # Add some seasonal/daily variation
        temp = b_temp + np.random.normal(0, s_temp, len(timestamps)) + np.sin(np.linspace(0, 50, len(timestamps)))
        humidity = b_hum + np.random.normal(0, s_hum, len(timestamps))
        
        # Clip humidity to 0-100
        humidity = np.clip(humidity, 0, 100)
        
        df_fac = pd.DataFrame({
            "timestamp": timestamps,
            "facility_type": facility,
            "temperature": temp,
            "humidity": humidity
        })
        
        # Introduce some missing values to test clean.py
        missing_idx = np.random.choice(df_fac.index, size=int(len(df_fac) * 0.05), replace=False)
        df_fac.loc[missing_idx, 'temperature'] = np.nan
        
        records.append(df_fac)
        
    final_df = pd.concat(records, ignore_index=True)
    
    # Shuffle a bit or just keep as is, we will partition it later anyway
    final_df.to_csv(output_path, index=False)
    print(f"Generated synthetic raw data at {output_path} with {len(final_df)} rows.")

if __name__ == "__main__":
    generate_raw_data()
