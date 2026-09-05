"""
Evaluation metrics computation for detection, behavior, and anomaly models.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def compute_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Computes Precision, Recall, F1 score, and Confusion Matrix.
    """
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "confusion_matrix": cm.tolist(),
    }


def compute_detection_ap(
    precisions: np.ndarray, recalls: np.ndarray
) -> float:
    """
    Computes Average Precision (AP) from precision-recall curve via 11-point interpolation or AUC.
    """
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return float(ap)
