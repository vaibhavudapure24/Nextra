"""
Observable stress index estimator for tracked wildlife animals.
"""

from typing import List, Dict, Any
import numpy as np


class StressEstimator:
    """
    Computes a normalized stress index [0.0 - 1.0] from observable motion agitation,
    erratic pacing, and velocity variance.
    """
    def estimate(
        self,
        speeds: List[float],
        running_duration_sec: float,
        aggression_duration_sec: float,
    ) -> float:
        if not speeds:
            return 0.0

        arr = np.array(speeds, dtype=np.float32)
        variance = float(np.var(arr)) if len(arr) > 1 else 0.0

        # Stress factors from kinematics
        speed_jitter_factor = min(0.4, variance * 0.1)
        running_factor = min(0.35, (running_duration_sec / 120.0) * 0.35)
        aggression_factor = min(0.35, (aggression_duration_sec / 30.0) * 0.35)

        raw_stress = speed_jitter_factor + running_factor + aggression_factor
        return float(np.clip(raw_stress, 0.0, 1.0))
