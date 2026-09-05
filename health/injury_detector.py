"""
Observable injury and limping detector based on gait asymmetry and motion periodicity.
"""

from typing import List, Dict, Any, Tuple
import numpy as np


class InjuryDetector:
    """
    Analyzes stride periodicity and motion asymmetry to estimate injury / limping risk.
    """
    def __init__(self, stride_variance_threshold: float = 0.35):
        self.stride_variance_threshold = stride_variance_threshold

    def evaluate_gait(
        self,
        speeds: List[float],
        aspect_ratios: List[float],
    ) -> Tuple[float, bool]:
        """
        Evaluates gait regularity from recent motion history.
        Returns (injury_risk_score [0.0 - 1.0], is_limping_detected).
        """
        if len(speeds) < 15:
            return 0.0, False

        # Measure stride asymmetry by alternating difference
        diffs = np.abs(np.diff(speeds[-15:]))
        if len(diffs) < 2:
            return 0.0, False

        odd_steps = diffs[0::2]
        even_steps = diffs[1::2]

        mean_odd = float(np.mean(odd_steps))
        mean_even = float(np.mean(even_steps))

        asymmetry = abs(mean_odd - mean_even) / (mean_odd + mean_even + 1e-4)

        risk_score = float(np.clip(asymmetry * 1.5, 0.0, 1.0))
        is_limping = bool(risk_score >= self.stride_variance_threshold)

        return round(risk_score, 3), is_limping
