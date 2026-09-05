"""
Trajectory management and movement heatmap generation.
"""

from typing import List, Tuple, Dict, Optional
import math
import numpy as np
import cv2
from utils.gps_utils import haversine_distance


class TrajectoryManager:
    """
    Maintains trajectory trails, calculates cumulative distance, and renders movement heatmaps.
    """
    def __init__(self, max_trail_points: int = 90, meters_per_pixel: float = 0.05):
        self.max_trail_points = max_trail_points
        self.meters_per_pixel = meters_per_pixel
        # Dict[track_id -> List of (x, y)]
        self.trajectories: Dict[int, List[Tuple[float, float]]] = {}
        # Dict[track_id -> float cumulative distance in meters]
        self.distances: Dict[int, float] = {}

    def update(self, track_id: int, centroid: Tuple[float, float], is_gps: bool = False) -> float:
        """
        Appends new centroid point and updates cumulative distance travelled.
        Returns total distance travelled in meters.
        """
        if track_id not in self.trajectories:
            self.trajectories[track_id] = [centroid]
            self.distances[track_id] = 0.0
            return 0.0

        pts = self.trajectories[track_id]
        last_pt = pts[-1]

        if is_gps:
            step_m = haversine_distance(last_pt[0], last_pt[1], centroid[0], centroid[1])
        else:
            step_px = math.hypot(centroid[0] - last_pt[0], centroid[1] - last_pt[1])
            step_m = step_px * self.meters_per_pixel

        self.distances[track_id] += step_m
        pts.append(centroid)

        if len(pts) > self.max_trail_points:
            pts.pop(0)

        return self.distances[track_id]

    def get_trail(self, track_id: int) -> List[Tuple[float, float]]:
        """Returns historical trail points for a given track ID."""
        return self.trajectories.get(track_id, [])

    def get_distance(self, track_id: int) -> float:
        """Returns total distance travelled in meters."""
        return self.distances.get(track_id, 0.0)

    def generate_heatmap(
        self,
        width: int,
        height: int,
        track_id: Optional[int] = None,
        radius: int = 15,
    ) -> np.ndarray:
        """
        Generates a 2D density heatmap of movement paths using Gaussian kernels.
        """
        density_map = np.zeros((height, width), dtype=np.float32)

        target_ids = [track_id] if track_id is not None else list(self.trajectories.keys())
        for tid in target_ids:
            pts = self.trajectories.get(tid, [])
            for px, py in pts:
                ix = int(px)
                iy = int(py)
                if 0 <= ix < width and 0 <= iy < height:
                    density_map[iy, ix] += 1.0

        if density_map.max() > 0:
            # Gaussian blur for smooth density field
            ksize = radius * 2 + 1
            density_map = cv2.GaussianBlur(density_map, (ksize, ksize), 0)
            # Normalize to 0-255
            norm_map = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX)
            norm_map = norm_map.astype(np.uint8)
            # Apply COLORMAP_JET
            heatmap_color = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
            return heatmap_color

        return np.zeros((height, width, 3), dtype=np.uint8)
