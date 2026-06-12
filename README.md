# Federated Learning IoT Platform

A full-stack, end-to-end Federated Learning architecture for anomaly detection in IoT telemetry networks. This platform allows distributed company nodes to train local LSTM Autoencoders on their private sensor data, and securely aggregates their model weights via a central Flower federation server to build a robust global model without ever sharing raw data.

---

## 🖥️ Platform UI Guide (Features & Buttons Explained)

The frontend interface is divided into two primary sections: The **Company Workspace** (for local node operations) and the **Admin Dashboard** (for triggering global federation).

### 1. Company Login Cards
**Location**: The Landing Page
* **Company A, B, C Cards**: These cards represent isolated edge nodes (e.g., individual companies or sensor networks). 
* **"Active Config" / "0 configured columns"**: Indicates whether that specific node has a validated schema ready for training.
* **Click Action**: Clicking a card opens the isolated workspace strictly for that company.

---

### 2. Dataset Upload Tab
**Location**: Inside a Company Workspace -> "Dataset Upload"

* **Drag & Drop Upload Box**: Accepts `.csv` files containing your raw IoT telemetry. When a file is dropped, it is securely uploaded to the node's local backend storage. Once uploaded, this box transforms into an **Active Dataset** banner.
* **Replace Dataset Button**: Allows you to upload a new `.csv` and overwrite the current node's active dataset.
* **Column Configuration Panel**: Automatically appears after upload. It parses your CSV headers and allows you to map your dataset to the LSTM Autoencoder.
  * **Timestamp Column Dropdown**: Select the column containing time data (or choose `<None>` to use row indexing).
  * **Label / Target Dropdown**: Select the ground-truth anomaly label if you have one (used only for calculating evaluation metrics, not for training the autoencoder).
  * **Feature Chips (Pills)**: Clickable buttons for every remaining column. **You must click the features you want the model to learn from.** Selected features highlight in blue.
* **Save Schema & Configuration Button**: Locks in your selected features and saves them to `config.json`. The Training tab will remain disabled until this is clicked.
* **Edit Schema Button**: Found in the green "Active Configuration" summary. Allows you to re-open the schema panel to add or remove feature columns without having to re-upload your CSV.

---

### 3. Local Model Training Tab
**Location**: Inside a Company Workspace -> "Train Model"

* **Epochs Slider**: Determine how many times the local LSTM Autoencoder will iterate over the dataset.
* **Sequence Window Slider**: The number of time-steps the LSTM will look backward to predict the next step. (e.g., a window of 24 means it looks at the last 24 sensor readings).
* **Batch Size Slider**: The number of sequences processed before the model updates its internal weights.
* **Start Local Training Button**: Compiles the Keras model, processes the sliding windows, and executes the training loop entirely on this specific node's data. 
* **Loss Curve Graph**: A dynamically rendered chart displaying the real Mean Squared Error (MSE) dropping over epochs as the model learns normal IoT behavior.

---

### 4. Anomaly Detection Tab
**Location**: Inside a Company Workspace -> "Detect Anomalies"

* **Anomaly Threshold (MSE) Slider**: Sets the strictness of the detector. If the model's reconstruction error for a specific time-window exceeds this value, it flags an anomaly. Lower values trigger more anomalies; higher values are more forgiving.
* **Run Local Detection Button**: Executes inference using the currently trained `.weights.h5` model. (This button is strictly disabled if no local model has been trained).
* **Detection Metrics Graph**: A ComposedChart mapping the true feature values (e.g., temperature) against red scatter dots representing time-steps that breached the MSE threshold.

---

### 5. Federated Admin Dashboard
**Location**: Top right navigation bar -> "Admin Dashboard"

* **Target Federation Rounds Slider**: Dictates how many global communication cycles the central server will execute. In each round, the server collects weights from all active companies, averages them, and redistributes the smarter global model back down to the nodes.
* **Trigger Federated Learning Button**: Bootstraps the Python `flwr` (Flower) server and connects the active company nodes as isolated subprocess clients.
* **Global Convergence Graph**: As federation progresses, this graph tracks the global aggregated loss. A downward trend proves that the federated model is effectively learning across all isolated datasets.
* **Status Console**: Real-time polling metrics displaying `rounds_completed` and the exact `Global_Model_vX.weights.h5` version currently deployed to the nodes.
