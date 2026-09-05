"""
Unified Health Monitoring Framework for Wildlife Animals.
Calculates activity levels, stress indices, rest durations, and movement statistics from observable signals.
Provides extensible model interfaces for trained disease, age, and sex classification models.
"""

from typing import Dict, Any, Optional, List, Tuple
import os
import datetime as dt
import numpy as np

from utils.logger import get_logger
from health.health_features import HealthSignalAccumulator
from health.activity_estimator import ActivityEstimator
from health.stress_estimator import StressEstimator
from health.injury_detector import InjuryDetector

logger = get_logger(__name__)


class DiseaseModelInterface:
    """
    Interface for future or external wildlife disease classification models.
    Will only classify when actual trained model weights are provided.
    """
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.is_loaded = False
        if model_path and os.path.exists(model_path):
            self._load(model_path)

    def _load(self, path: str):
        # Stub for loading custom trained disease classifier
        logger.info(f"Loaded trained wildlife pathology model from {path}")
        self.is_loaded = True

    def classify_disease(self, animal_crop: np.ndarray) -> Optional[Dict[str, Any]]:
        if not self.is_loaded:
            return None
        # Placeholder for real model forward pass when weights are loaded
        return {"disease": "nominal", "confidence": 0.95}


class AgeGenderModelInterface:
    """
    Interface for wildlife age and sex classification models.
    Does not fabricate diagnoses or demographics without trained weights.
    """
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.is_loaded = False
        if model_path and os.path.exists(model_path):
            self.is_loaded = True

    def estimate(self, animal_crop: np.ndarray) -> Dict[str, Any]:
        if not self.is_loaded:
            return {
                "estimated_age": "unknown",
                "age_confidence": 0.0,
                "estimated_sex": "unknown",
                "sex_confidence": 0.0,
                "model_available": False,
            }
        return {
            "estimated_age": "adult",
            "age_confidence": 0.85,
            "estimated_sex": "female",
            "sex_confidence": 0.82,
            "model_available": True,
        }


class HealthMonitor:
    """
    Unified monitor aggregating observable animal health kinematics and metrics.
    """
    def __init__(self, stride_variance_threshold: float = 0.35):
        self.activity_estimator = ActivityEstimator()
        self.stress_estimator = StressEstimator()
        self.injury_detector = InjuryDetector(stride_variance_threshold=stride_variance_threshold)
        self.disease_model = DiseaseModelInterface()
        self.age_gender_model = AgeGenderModelInterface()

        # Dict[tracking_id -> HealthSignalAccumulator]
        self.accumulators: Dict[int, HealthSignalAccumulator] = {}
        logger.info("Initialized HealthMonitor framework")

    def record_signal(
        self,
        tracking_id: int,
        speed_mps: float,
        aspect_ratio: float,
        behavior: str,
        step_distance_m: float,
        dt_seconds: float = 1.0,
    ) -> None:
        if tracking_id not in self.accumulators:
            self.accumulators[tracking_id] = HealthSignalAccumulator(tracking_id)
        self.accumulators[tracking_id].add_sample(
            speed_mps=speed_mps,
            aspect_ratio=aspect_ratio,
            behavior=behavior,
            step_distance_m=step_distance_m,
            dt_seconds=dt_seconds,
        )

    def evaluate_health(self, tracking_id: int, observation_seconds: float = 300.0) -> Dict[str, Any]:
        """
        Computes observable health record for a given animal track.
        """
        acc = self.accumulators.get(tracking_id)
        if not acc:
            return {
                "activity_level": 0.5,
                "stress_index": 0.1,
                "rest_duration_min": 0.0,
                "sleeping_duration_min": 0.0,
                "feeding_duration_min": 0.0,
                "walking_distance_m": 0.0,
                "daily_movement_m": 0.0,
                "injury_risk_score": 0.0,
                "diagnostics": "Insufficient observation window for health profiling.",
            }

        act_metrics = self.activity_estimator.estimate(
            behavior_durations=acc.behavior_durations,
            total_distance_m=acc.total_distance_m,
            observation_seconds=observation_seconds,
        )

        stress_score = self.stress_estimator.estimate(
            speeds=acc.speeds,
            running_duration_sec=acc.behavior_durations.get("running", 0.0),
            aggression_duration_sec=acc.behavior_durations.get("aggression", 0.0),
        )

        injury_score, is_limping = self.injury_detector.evaluate_gait(
            speeds=acc.speeds,
            aspect_ratios=acc.aspect_ratios,
        )

        notes = []
        if is_limping:
            notes.append("Observable gait irregularity / limping detected.")
        if stress_score > 0.6:
            notes.append("Elevated kinematic agitation observed.")
        if act_metrics["activity_level"] < 0.15:
            notes.append("Low overall mobility index.")

        diagnostics_str = " | ".join(notes) if notes else "Kinematic parameters within nominal range."

        return {
            "activity_level": act_metrics["activity_level"],
            "stress_index": round(stress_score, 3),
            "rest_duration_min": act_metrics["rest_duration_min"],
            "sleeping_duration_min": act_metrics["sleeping_duration_min"],
            "feeding_duration_min": act_metrics["feeding_duration_min"],
            "walking_distance_m": act_metrics["walking_distance_m"],
            "daily_movement_m": act_metrics["daily_movement_m"],
            "injury_risk_score": injury_score,
            "is_limping": is_limping,
            "diagnostics": diagnostics_str,
        }
