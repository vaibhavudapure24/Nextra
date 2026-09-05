"""
Isolation Forest anomaly detection wrapper using scikit-learn.
"""

from typing import Optional
import os
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from utils.logger import get_logger

logger = get_logger(__name__)


class IsolationForestDetector:
    """
    Isolation Forest model scoring anomalies via path lengths in random partition trees.
    """
    def __init__(self, contamination: float = 0.05, n_estimators: int = 100):
        self.contamination = contamination
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42,
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray) -> None:
        self.model.fit(X)
        self.is_fitted = True

    def score(self, feat: np.ndarray) -> float:
        """
        Returns normalized anomaly score between 0.0 (normal) and 1.0 (anomalous).
        """
        if not self.is_fitted:
            # Fallback baseline calculation when untrained
            return 0.1
        if feat.ndim == 1:
            feat = feat.reshape(1, -1)
        # decision_function gives negative score for outliers, positive for inliers
        raw = self.model.decision_function(feat)[0]
        # Map raw decision function [-0.5, 0.5] into [0, 1] anomaly score
        score = 1.0 / (1.0 + np.exp(raw * 8.0))
        return float(np.clip(score, 0.0, 1.0))

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path: str) -> bool:
        if os.path.exists(path):
            self.model = joblib.load(path)
            self.is_fitted = True
            return True
        return False
