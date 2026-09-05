"""
Postprocessing data structures and operations for animal detection.
Standardizes bounding boxes, confidence filtering, and detection metadata.
"""

from __future__ import annotations
import uuid
import datetime as dt
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
import numpy as np


@dataclass
class DetectionResult:
    """
    Standardized detection output containing all spatial and temporal attributes.
    """
    detection_id: str
    class_id: int
    species: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    center_x: float
    center_y: float
    frame_number: int = 0
    timestamp: float = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).timestamp())
    tracking_id: Optional[int] = None

    @classmethod
    def create(
        cls,
        class_id: int,
        species: str,
        confidence: float,
        bbox: Tuple[float, float, float, float],
        frame_number: int = 0,
        timestamp: Optional[float] = None,
        tracking_id: Optional[int] = None,
        detection_id: Optional[str] = None,
    ) -> DetectionResult:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        ts = timestamp if timestamp is not None else dt.datetime.now(dt.timezone.utc).timestamp()
        det_id = detection_id or f"det_{uuid.uuid4().hex[:8]}"
        return cls(
            detection_id=det_id,
            class_id=class_id,
            species=species,
            confidence=float(confidence),
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            center_x=cx,
            center_y=cy,
            frame_number=frame_number,
            timestamp=ts,
            tracking_id=tracking_id,
        )

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detection_id": self.detection_id,
            "class_id": self.class_id,
            "species": self.species,
            "confidence": round(self.confidence, 4),
            "x1": round(self.x1, 2),
            "y1": round(self.y1, 2),
            "x2": round(self.x2, 2),
            "y2": round(self.y2, 2),
            "center_x": round(self.center_x, 2),
            "center_y": round(self.center_y, 2),
            "frame_number": self.frame_number,
            "timestamp": self.timestamp,
            "tracking_id": self.tracking_id,
        }


def filter_by_confidence(
    detections: List[DetectionResult], min_confidence: float
) -> List[DetectionResult]:
    """Filters detection list by confidence threshold."""
    return [d for d in detections if d.confidence >= min_confidence]


def compute_iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes (x1, y1, x2, y2)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    boxBArea = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])

    unionArea = boxAArea + boxBArea - interArea
    if unionArea <= 0:
        return 0.0
    return interArea / unionArea
