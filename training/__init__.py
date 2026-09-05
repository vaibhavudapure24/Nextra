"""
Wildlife Model Training and Evaluation Pipeline.
"""

from training.metrics import compute_classification_metrics, compute_detection_ap
from training.train_detection import train_detection
from training.prepare_dataset import prepare_yolo_dataset, validate_yolo_label_file
from training.train_behavior import train_behavior_model
from training.train_anomaly import train_anomaly_models
from training.train_health import train_health_predictor

__all__ = [
    "compute_classification_metrics",
    "compute_detection_ap",
    "train_detection",
    "prepare_yolo_dataset",
    "validate_yolo_label_file",
    "train_behavior_model",
    "train_anomaly_models",
    "train_health_predictor",
]
