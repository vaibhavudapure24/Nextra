"""
Feature extraction for temporal animal behavior recognition.
Computes kinematics: position, velocity, acceleration, heading, bounding box aspect ratio,
movement frequency, and inter-animal social distance.
"""

from typing import List, Tuple, Optional, Dict
import math
import numpy as np


FEATURE_DIM = 12


def extract_frame_features(
    curr_pos: Tuple[float, float],
    prev_pos: Optional[Tuple[float, float]],
    prev_prev_pos: Optional[Tuple[float, float]],
    bbox: Tuple[float, float, float, float],
    dt: float = 1.0 / 30.0,
    neighbor_positions: Optional[List[Tuple[float, float]]] = None,
) -> np.ndarray:
    """
    Extracts a 12-dimensional feature vector for a single frame:
    [0]: normalized center x
    [1]: normalized center y
    [2]: normalized width
    [3]: normalized height
    [4]: aspect ratio (w / h)
    [5]: velocity magnitude (px/s)
    [6]: acceleration magnitude (px/s^2)
    [7]: heading angle (radians, [-pi, pi])
    [8]: bounding box area change rate (dw * dh / dt)
    [9]: vertical movement ratio (abs(dy) / (abs(dx) + abs(dy) + eps))
    [10]: minimum social distance to nearest tracked neighbor (normalized)
    [11]: movement frequency / motion jitter
    """
    x1, y1, x2, y2 = bbox
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    cx, cy = curr_pos
    aspect_ratio = w / h

    # Velocity
    if prev_pos is not None and dt > 0:
        vx = (cx - prev_pos[0]) / dt
        vy = (cy - prev_pos[1]) / dt
        speed = math.hypot(vx, vy)
        heading = math.atan2(vy, vx)
        dy = abs(cy - prev_pos[1])
        dx = abs(cx - prev_pos[0])
        vertical_ratio = dy / (dx + dy + 1e-6)
    else:
        vx, vy = 0.0, 0.0
        speed = 0.0
        heading = 0.0
        vertical_ratio = 0.0

    # Acceleration
    if prev_prev_pos is not None and prev_pos is not None and dt > 0:
        prev_vx = (prev_pos[0] - prev_prev_pos[0]) / dt
        prev_vy = (prev_pos[1] - prev_prev_pos[1]) / dt
        ax = (vx - prev_vx) / dt
        ay = (vy - prev_vy) / dt
        acceleration = math.hypot(ax, ay)
    else:
        acceleration = 0.0

    # Social interaction distance
    min_social_dist = 1.0  # default far
    if neighbor_positions:
        distances = [math.hypot(cx - nx, cy - ny) for nx, ny in neighbor_positions]
        if distances:
            # normalized by rough frame dimension 1280
            min_social_dist = min(distances) / 1280.0

    # Motion frequency / jitter proxy
    jitter = math.log1p(speed * (1.0 + acceleration * 0.01))

    feat = np.array([
        cx / 1280.0,
        cy / 720.0,
        w / 1280.0,
        h / 720.0,
        aspect_ratio,
        speed / 1000.0,
        acceleration / 5000.0,
        heading / math.pi,
        (w * h) / (1280.0 * 720.0),
        vertical_ratio,
        min(1.0, min_social_dist),
        min(1.0, jitter / 10.0),
    ], dtype=np.float32)

    return feat
