# Federated Learning - P1 & P2 Module

## Overview
This module implements an LSTM Autoencoder for anomaly detection in IoT time-series data.

It is designed to support federated learning where each device acts as a client.

---

## Components

- model_pipeline.py → model + training
- client_data.py → device → client mapping
- detection_utils.py → anomaly grouping (optional)
- config.yaml → parameters

---

## Input Shape
(batch_size, 24, num_features)

---

## Federated Learning Usage

Each client:
1. preprocess data
2. train model locally
3. send weights using get_weights()
4. receive updated weights using set_weights()

---

## Detection (Optional)

Reconstruction error is used.

Default:
- threshold = 0.03
- gap = 300

---

## Notes

- Data is non-IID across devices
- Detection is separate from training
- Compatible with Flower