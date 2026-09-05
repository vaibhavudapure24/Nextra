"""
Base Animal Tracker interface and TrackedObject data representation.
"""

from __future__ import annotations
import abc
import datetime as dt
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np

from utils.logger import get_logger
from utils.image import draw_bounding_box, draw_trajectory, get_color_for_id
from detection.postprocessing import DetectionResult

logger = get_logger(__name__)


@dataclass
class TrackedObject:
    """
    Representation of an actively tracked animal across frames.
    """
    tracking_id: int
    species: str
    first_seen: float
    last_seen: float
    current_position: Tuple[float, float]
    previous_position: Tuple[float, float]
    bbox: Tuple[float, float, float, float]
    confidence: float
    speed: float = 0.0                      # instantaneous speed in m/s
    distance_travelled: float = 0.0         # cumulative distance in meters
    trajectory: List[Tuple[float, float]] = field(default_factory=list)
    time_since_update: int = 0
    hit_streak: int = 0
    class_id: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tracking_id": self.tracking_id,
            "species": self.species,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "current_position": [round(v, 2) for v in self.current_position],
            "previous_position": [round(v, 2) for v in self.previous_position],
            "bbox": [round(v, 2) for v in self.bbox],
            "confidence": round(self.confidence, 4),
            "speed": round(self.speed, 2),
            "distance_travelled": round(self.distance_travelled, 2),
            "trajectory_points": len(self.trajectory),
        }


class BaseTracker(abc.ABC):
    """
    Abstract interface for object tracking algorithms.
    """
    def __init__(self, max_trail_length: int = 90):
        self.max_trail_length = max_trail_length
        self.active_tracks: Dict[int, TrackedObject] = {}

    @abc.abstractmethod
    def update(
        self,
        detections: List[DetectionResult],
        frame: Optional[np.ndarray] = None,
        frame_number: int = 0,
        timestamp: Optional[float] = None,
    ) -> List[TrackedObject]:
        """
        Updates tracker state with new frame detections.
        """
        pass

    def draw(
        self,
        frame: np.ndarray,
        tracks: List[TrackedObject],
        draw_trails: bool = True,
        show_speed: bool = True,
    ) -> np.ndarray:
        """
        Draws bounding boxes, tracking IDs, speed, and motion trails on the frame.
        """
        annotated = frame.copy()
        for trk in tracks:
            color = get_color_for_id(trk.tracking_id)
            speed_txt = f" | {trk.speed:.1f} m/s" if (show_speed and trk.speed > 0) else ""
            label = f"#{trk.tracking_id} {trk.species} ({trk.confidence*100:.0f}%){speed_txt}"

            annotated = draw_bounding_box(annotated, trk.bbox, label, color=color, thickness=2)

            if draw_trails and trk.trajectory:
                annotated = draw_trajectory(annotated, trk.trajectory, color=color, thickness=2)

        return annotated


def get_tracker(
    tracker_type: str = "bytetrack",
    cfg=None,
) -> BaseTracker:
    """
    Factory function returning ByteTrack or DeepSORT tracker instance.
    """
    t_type = tracker_type.lower().strip()
    if t_type == "deepsort":
        from tracking.deepsort_tracker import DeepSORTTracker
        return DeepSORTTracker()
    else:
        from tracking.bytetrack_tracker import ByteTrackTracker
        return ByteTrackTracker()
