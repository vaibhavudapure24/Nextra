"""
Feature extraction for anomaly detection in wildlife monitoring.
Constructs multi-dimensional kinematic vectors evaluating speed variance, sudden posture shifts,
stillness duration, and social distance.
"""

from typing import List, Tuple, Dict, Any, Optional
import math
import numpy as np


def extract_anomaly_features(
    speed_history: List[float],
    aspect_ratios: List[float],
    distance_to_fence_m: float,
    nearest_neighbor_distance_m: float,
    inactivity_seconds: float = 0.0,
    running_seconds: float = 0.0,
) -> np.ndarray:
    """
    Extracts an 8-dimensional feature vector for anomaly models:
    [0]: mean speed (m/s)
    [1]: speed variance / volatility
    [2]: max acceleration proxy (max speed - min speed)
    [3]: aspect ratio rate of change (sudden collapse / fall proxy)
    [4]: continuous stillness / inactivity duration in minutes
    [5]: continuous running duration in minutes
    [6]: normalized distance to fence boundary (1.0 = far inside, 0.0 = at or past fence)
    [7]: isolation distance to herd / companions (meters)
    """
    if not speed_history:
        speed_history = [0.0]
    if not aspect_ratios:
        aspect_ratios = [1.0]

    speeds = np.array(speed_history, dtype=np.float32)
    ars = np.array(aspect_ratios, dtype=np.float32)

    mean_speed = float(np.mean(speeds))
    speed_var = float(np.var(speeds)) if len(speeds) > 1 else 0.0
    speed_range = float(np.max(speeds) - np.min(speeds))

    # Aspect ratio shift
    ar_change = float(abs(ars[-1] - ars[0])) if len(ars) > 1 else 0.0

    feat = np.array([
        mean_speed,
        speed_var,
        speed_range,
        ar_change,
        inactivity_seconds / 60.0,
        running_seconds / 60.0,
        max(0.0, min(1.0, distance_to_fence_m / 100.0)),
        min(100.0, nearest_neighbor_distance_m),
    ], dtype=np.float32)

    return feat
