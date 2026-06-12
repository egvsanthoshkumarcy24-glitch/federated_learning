import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class NodeDataShard:
    """Format 2: Node data shard format"""
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray

@dataclass
class FLMetrics:
    """Format 4: Metrics dict structure for evaluation"""
    f1_score: float
    precision: float
    recall: float
    auc_roc: float
    mse_threshold: float

# Format 3: model weight format for Flower is typically List[np.ndarray]
FlowerWeights = List[np.ndarray]
