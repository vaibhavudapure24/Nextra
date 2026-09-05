"""
Speed estimation module for tracked wildlife animals.
Calculates instantaneous and smoothed speed in meters/second and km/h.
"""

from typing import Tuple, Optional
import math
from utils.gps_utils import haversine_distance


class SpeedEstimator:
    """
    Estimates movement speed of tracked animals from pixel motion or GPS positions.
    """
    def __init__(
        self,
        meters_per_pixel: float = 0.05,
        fps: float = 30.0,
        smoothing_factor: float = 0.8,
    ):
        self.meters_per_pixel = meters_per_pixel
        self.fps = fps if fps > 0 else 30.0
        self.smoothing_factor = smoothing_factor

    def estimate_speed(
        self,
        pos_prev: Tuple[float, float],
        pos_curr: Tuple[float, float],
        time_delta_sec: Optional[float] = None,
        previous_speed: float = 0.0,
        is_gps: bool = False,
    ) -> float:
        """
        Estimates instantaneous speed in meters/second.
        """
        dt = time_delta_sec if (time_delta_sec is not None and time_delta_sec > 0) else (1.0 / self.fps)

        if is_gps:
            dist_meters = haversine_distance(pos_prev[0], pos_prev[1], pos_curr[0], pos_curr[1])
        else:
            dx = pos_curr[0] - pos_prev[0]
            dy = pos_curr[1] - pos_prev[1]
            dist_pixels = math.hypot(dx, dy)
            dist_meters = dist_pixels * self.meters_per_pixel

        raw_speed = dist_meters / dt if dt > 0 else 0.0

        # Apply exponential moving average filter
        if previous_speed <= 0.0:
            return raw_speed
        return self.smoothing_factor * previous_speed + (1.0 - self.smoothing_factor) * raw_speed

    @staticmethod
    def mps_to_kmh(speed_mps: float) -> float:
        """Converts meters/second to kilometers/hour."""
        return speed_mps * 3.6
