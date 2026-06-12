import requests
import json
import time

BASE_URL = "http://localhost:8000/api"

print("--- 1. UPLOAD ---")
files = {'file': open('datasets/cold_chain/synthetic_iot_dataset_challenging.csv', 'rb')}
r_upload = requests.post(f"{BASE_URL}/upload/Company_A", files=files)
print(r_upload.json())

print("\n--- 2. CONFIG ---")
cfg = {
    "timestamp_col": "Timestamp",
    "feature_cols": ["Temperature", "Humidity"],
    "label_col": "Anomaly"
}
r_config = requests.post(f"{BASE_URL}/config/Company_A", json=cfg)
print(r_config.json())

print("\n--- 3. TRAIN ---")
r_train = requests.post(f"{BASE_URL}/train/Company_A", json={"epochs": 2, "window_size": 24, "batch_size": 32})
print("Training Status:", r_train.json().get('status'))
hist = r_train.json().get('history', [])
print(f"Epochs returned: {len(hist)}")
if len(hist) > 0:
    print(f"First Epoch Loss: {hist[0]}")
    print(f"Last Epoch Loss: {hist[-1]}")

print("\n--- 4. DETECT ---")
r_detect = requests.post(f"{BASE_URL}/detect/Company_A", json={"threshold": 0.05})
det_json = r_detect.json()
print("Detection Status:", det_json.get('status'))
print("Total Anomalies:", det_json.get('total_anomalies'))
print("Total Scanned:", det_json.get('total_scanned'))

print("\n--- 5. SETUP COMPANY B ---")
files2 = {'file': open('datasets/cold_chain/synthetic_iot_dataset_challenging.csv', 'rb')}
requests.post(f"{BASE_URL}/upload/Company_B", files=files2)
requests.post(f"{BASE_URL}/config/Company_B", json=cfg)
print("Company B Setup Complete")

print("\n--- 6. FEDERATION TRIGGER ---")
r_fed = requests.post(f"{BASE_URL}/federate", json={"num_rounds": 2, "window_size": 24, "n_features": 2})
print("Federation Trigger Status:", r_fed.json())

print("Waiting for Federation to complete (15s)...")
time.sleep(15)

print("\n--- 7. DASHBOARD METRICS ---")
r_stat = requests.get(f"{BASE_URL}/federate/status")
print(r_stat.json())
