"""
One-Class Support Vector Machine (SVM) anomaly detector using scikit-learn.
"""

from typing import Optional
import os
import joblib
import numpy as np
from sklearn.svm import OneClassSVM
from utils.logger import get_logger

logger = get_logger(__name__)


class OneClassSVMDetector:
    """
    One-Class SVM modeling high-density nominal behavior support boundary.
    """
    def __init__(self, nu: float = 0.05, kernel: str = "rbf", gamma: str = "scale"):
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
        self.is_fitted = False

    def fit(self, X: np.ndarray) -> None:
        self.model.fit(X)
        self.is_fitted = True

    def score(self, feat: np.ndarray) -> float:
        """
        Returns normalized anomaly score between 0.0 (normal) and 1.0 (anomalous).
        """
        if not self.is_fitted:
            return 0.1
        if feat.ndim == 1:
            feat = feat.reshape(1, -1)
        raw = self.model.decision_function(feat)[0]
        score = 1.0 / (1.0 + np.exp(raw * 5.0))
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
