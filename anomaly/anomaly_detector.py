"""
Unified Anomaly Detection coordinator for Wildlife Animal Monitoring.
Integrates Isolation Forest, One-Class SVM, and Autoencoder methods with
domain rule heuristics for wildlife safety.
"""

from typing import Dict, Any, Optional, List, Tuple
import time
import datetime as dt
import numpy as np

from utils.logger import get_logger
from database.models import AlertSeverity
from anomaly.anomaly_features import extract_anomaly_features
from anomaly.isolation_forest import IsolationForestDetector
from anomaly.one_class_svm import OneClassSVMDetector
from anomaly.autoencoder import AnomalyAutoencoder, load_autoencoder

logger = get_logger(__name__)


class AnomalyDetector:
    """
    Coordinates anomaly detection models and categorizes specific wildlife emergencies.
    """
    def __init__(
        self,
        method: str = "isolation_forest",
        score_threshold: float = 0.70,
        iforest_path: str = "models/anomaly/isolation_forest.joblib",
        svm_path: str = "models/anomaly/one_class_svm.joblib",
        autoencoder_path: str = "models/anomaly/autoencoder.pt",
    ):
        self.method = method.lower()
        self.score_threshold = score_threshold

        self.iforest = IsolationForestDetector()
        self.iforest.load(iforest_path)

        self.svm = OneClassSVMDetector()
        self.svm.load(svm_path)

        self.autoencoder = load_autoencoder(autoencoder_path)

        logger.info(f"Initialized AnomalyDetector (active method: {self.method})")

    def evaluate(
        self,
        tracking_id: int,
        speed_history: List[float],
        aspect_ratios: List[float],
        distance_to_fence_m: float = 100.0,
        nearest_neighbor_distance_m: float = 10.0,
        inactivity_seconds: float = 0.0,
        running_seconds: float = 0.0,
        behavior_type: str = "walking",
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluates track kinematics against statistical models and domain triggers.
        Returns anomaly dictionary if an anomaly condition is triggered, else None.
        """
        ts = dt.datetime.now(dt.timezone.utc).timestamp()
        feat = extract_anomaly_features(
            speed_history=speed_history,
            aspect_ratios=aspect_ratios,
            distance_to_fence_m=distance_to_fence_m,
            nearest_neighbor_distance_m=nearest_neighbor_distance_m,
            inactivity_seconds=inactivity_seconds,
            running_seconds=running_seconds,
        )

        # 1. Evaluate specific high-priority domain patterns
        specific_anomaly = self._check_domain_patterns(
            feat=feat,
            distance_to_fence_m=distance_to_fence_m,
            inactivity_seconds=inactivity_seconds,
            running_seconds=running_seconds,
            behavior_type=behavior_type,
            ts=ts,
        )
        if specific_anomaly is not None:
            return specific_anomaly

        # 2. Evaluate statistical / neural anomaly score
        model_score = self._compute_model_score(feat)
        if model_score >= self.score_threshold:
            severity = AlertSeverity.HIGH if model_score > 0.85 else AlertSeverity.MEDIUM
            return {
                "anomaly_score": round(model_score, 4),
                "anomaly_type": "ABNORMAL_BEHAVIOR",
                "severity": severity,
                "confidence": 0.80,
                "timestamp": ts,
                "details": f"Kinematic anomaly detected by {self.method} (score={model_score:.3f})",
            }

        return None

    def _compute_model_score(self, feat: np.ndarray) -> float:
        if self.method == "one_class_svm" and self.svm.is_fitted:
            return self.svm.score(feat)
        elif self.method == "autoencoder" and self.autoencoder is not None:
            import torch
            tensor = torch.from_numpy(feat).float()
            return self.autoencoder.compute_anomaly_score(tensor)
        elif self.iforest.is_fitted:
            return self.iforest.score(feat)

        # Heuristic baseline score when models are not yet trained with custom dataset
        # Evaluates variance of speed and sudden aspect ratio jump
        raw_proxy = (feat[1] * 0.2) + (feat[3] * 0.4) + (feat[4] * 0.05)
        return float(np.clip(raw_proxy, 0.0, 0.65))

    def _check_domain_patterns(
        self,
        feat: np.ndarray,
        distance_to_fence_m: float,
        inactivity_seconds: float,
        running_seconds: float,
        behavior_type: str,
        ts: float,
    ) -> Optional[Dict[str, Any]]:
        # Fence escape: distance <= 0 meters or past fence
        if distance_to_fence_m <= 0.0:
            return {
                "anomaly_score": 0.99,
                "anomaly_type": "FENCE_ESCAPE",
                "severity": AlertSeverity.CRITICAL,
                "confidence": 0.95,
                "timestamp": ts,
                "details": "Animal crossed the virtual enclosure perimeter boundary.",
            }

        # Animal fall: sudden large aspect ratio collapse while moving
        ar_change = feat[3]
        if ar_change > 1.2 and feat[0] > 0.5:
            return {
                "anomaly_score": 0.92,
                "anomaly_type": "ANIMAL_FALL",
                "severity": AlertSeverity.HIGH,
                "confidence": 0.88,
                "timestamp": ts,
                "details": "Sudden posture collapse / rapid vertical shift detected.",
            }

        # Prolonged inactivity: > 180 seconds motionless outside sleeping/resting
        if inactivity_seconds > 180.0 and behavior_type not in ("sleeping", "resting"):
            return {
                "anomaly_score": 0.82,
                "anomaly_type": "PROLONGED_INACTIVITY",
                "severity": AlertSeverity.MEDIUM,
                "confidence": 0.85,
                "timestamp": ts,
                "details": f"Animal motionless for {inactivity_seconds:.0f}s without resting.",
            }

        # Continuous frantic running: > 60 seconds
        if running_seconds > 60.0:
            return {
                "anomaly_score": 0.88,
                "anomaly_type": "CONTINUOUS_RUNNING",
                "severity": AlertSeverity.HIGH,
                "confidence": 0.90,
                "timestamp": ts,
                "details": f"Continuous high-speed running sustained for {running_seconds:.0f}s.",
            }

        # Abnormal aggression
        if behavior_type in ("aggression", "aggressive"):
            return {
                "anomaly_score": 0.85,
                "anomaly_type": "AGGRESSION",
                "severity": AlertSeverity.HIGH,
                "confidence": 0.82,
                "timestamp": ts,
                "details": "Aggressive behavior detected between animals or towards boundary.",
            }

        # Medical emergency / Injury
        if behavior_type == "injury":
            return {
                "anomaly_score": 0.95,
                "anomaly_type": "MEDICAL_EMERGENCY",
                "severity": AlertSeverity.CRITICAL,
                "confidence": 0.88,
                "timestamp": ts,
                "details": "Severe gait asymmetry or trauma indicator observed.",
            }

        return None
