import os
import pandas as pd

def clean_data(input_path="data/raw/synthetic_cold_storage.csv", output_path="data/processed/clean_data.csv"):
    if not os.path.exists(input_path):
        print(f"File {input_path} not found. Please run generate_synthetic_data.py first.")
        return
        
    df = pd.read_csv(input_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort by facility and time
    df = df.sort_values(by=['facility_type', 'timestamp'])
    
    # Handle missing values (forward-fill for sensor gaps <= 2hrs)
    # pandas ffill with limit
    df['temperature'] = df.groupby('facility_type')['temperature'].ffill(limit=2)
    df['humidity'] = df.groupby('facility_type')['humidity'].ffill(limit=2)
    
    # Drop remaining NaNs
    df = df.dropna(subset=['temperature', 'humidity'])
    
    # Normalize humidity to 0-100% just in case
    df['humidity'] = df['humidity'].clip(0, 100)
    
    # Remove duplicate timestamps per facility
    df = df.drop_duplicates(subset=['facility_type', 'timestamp'])
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path} with {len(df)} rows.")

if __name__ == "__main__":
    clean_data()
