"""
Health features and rolling metric aggregations for wildlife monitoring.
"""

from typing import List, Dict, Any, Optional
import numpy as np


class HealthSignalAccumulator:
    """
    Accumulates observable kinematic and behavioral signals across time slices
    for a specific animal tracking identity.
    """
    def __init__(self, tracking_id: int):
        self.tracking_id = tracking_id
        self.speeds: List[float] = []
        self.aspect_ratios: List[float] = []
        self.behavior_durations: Dict[str, float] = {
            "resting": 0.0,
            "sleeping": 0.0,
            "eating": 0.0,
            "walking": 0.0,
            "running": 0.0,
            "playing": 0.0,
            "limping": 0.0,
            "aggression": 0.0,
            "stress": 0.0,
        }
        self.total_distance_m: float = 0.0

    def add_sample(
        self,
        speed_mps: float,
        aspect_ratio: float,
        behavior: str,
        step_distance_m: float,
        dt_seconds: float = 1.0,
    ) -> None:
        self.speeds.append(speed_mps)
        self.aspect_ratios.append(aspect_ratio)
        self.total_distance_m += step_distance_m

        norm_beh = behavior.lower()
        if norm_beh in self.behavior_durations:
            self.behavior_durations[norm_beh] += dt_seconds
        else:
            self.behavior_durations["walking"] += dt_seconds
