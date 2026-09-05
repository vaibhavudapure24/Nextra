"""
Base Animal Detector interface and factory for Wildlife Animal Monitoring.
Supports YOLO11, YOLOv8, and custom fine-tuned weights with CUDA/CPU/ONNX acceleration.
"""

from __future__ import annotations
import os
import time
import abc
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np

from utils.logger import get_logger
from utils.device import select_device
from utils.config_loader import load_config, load_species_map, resolve_path
from utils.image import draw_bounding_box, get_color_for_id
from detection.postprocessing import DetectionResult

logger = get_logger(__name__)


class BaseDetector(abc.ABC):
    """
    Abstract interface for all wildlife detection models.
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.40,
        iou_threshold: float = 0.50,
        device: str = "auto",
        species_map_path: Optional[str] = None,
    ):
        self.model_path = model_path or "yolo11n.pt"
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device_str = device
        self.torch_device = select_device(device)
        self.species_map_path = species_map_path or "configs/species_map.yaml"
        self.species_map = self._load_species_map()
        self.model = None
        self._load_model()

    @abc.abstractmethod
    def _load_model(self) -> None:
        """Loads model weights to the target device."""
        pass

    @abc.abstractmethod
    def detect(
        self,
        frame: np.ndarray,
        confidence_threshold: Optional[float] = None,
        frame_number: int = 0,
        timestamp: Optional[float] = None,
    ) -> List[DetectionResult]:
        """
        Executes inference on an input frame and returns standardized DetectionResults.
        """
        pass

    def _load_species_map(self) -> Dict[int, str]:
        """Loads class ID to species name mapping."""
        resolved = resolve_path(self.species_map_path)
        if not os.path.exists(resolved):
            logger.warning(f"Species map not found at {resolved}, using model class names.")
            return {}
        try:
            cfg = load_species_map(resolved)
            mapping = {}
            for section in ("coco_pretrained", "custom_wildlife"):
                entries = cfg.get(section) or {}
                for k, v in entries.items():
                    mapping[int(k)] = str(v)
            return mapping
        except Exception as e:
            logger.warning(f"Failed to parse species map: {e}")
            return {}

    def get_species_name(self, class_id: int, fallback_name: str = "") -> str:
        """Maps numeric class ID to friendly species name."""
        return self.species_map.get(class_id, fallback_name or f"animal_{class_id}")

    def draw(
        self,
        frame: np.ndarray,
        detections: List[DetectionResult],
        show_tracking_id: bool = True,
    ) -> np.ndarray:
        """
        Renders bounding boxes, species name, confidence, and tracking ID on the frame.
        """
        annotated = frame.copy()
        for det in detections:
            color = get_color_for_id(det.tracking_id if det.tracking_id is not None else det.class_id)
            track_label = f" #{det.tracking_id}" if (show_tracking_id and det.tracking_id is not None) else ""
            label = f"{det.species}{track_label} {det.confidence * 100:.1f}%"
            annotated = draw_bounding_box(
                annotated,
                (det.x1, det.y1, det.x2, det.y2),
                label=label,
                color=color,
                thickness=2,
            )
        return annotated


def get_detector(
    cfg=None,
    model_type: Optional[str] = None,
    model_path: Optional[str] = None,
    confidence_threshold: Optional[float] = None,
    device: Optional[str] = None,
) -> BaseDetector:
    """
    Factory method to instantiate the appropriate detector based on configuration.
    """
    config = cfg or load_config()
    det_cfg = config.detection

    m_type = (model_type or os.getenv("MODEL_TYPE") or det_cfg.model_type).lower()
    m_path = model_path or os.getenv("MODEL_PATH") or det_cfg.model_path
    conf = confidence_threshold or float(os.getenv("CONFIDENCE_THRESHOLD", det_cfg.confidence_threshold))
    iou = float(os.getenv("IOU_THRESHOLD", det_cfg.iou_threshold))
    dev = device or os.getenv("DEVICE", det_cfg.device)

    # Import specialized detectors
    if "v8" in m_type:
        from detection.yolov8_detector import YOLOv8Detector
        return YOLOv8Detector(
            model_path=m_path,
            confidence_threshold=conf,
            iou_threshold=iou,
            device=dev,
            species_map_path=det_cfg.species_map_path,
        )
    else:
        # Default to YOLO11
        from detection.yolo11_detector import YOLO11Detector
        return YOLO11Detector(
            model_path=m_path,
            confidence_threshold=conf,
            iou_threshold=iou,
            device=dev,
            species_map_path=det_cfg.species_map_path,
        )


# Compatibility alias
AnimalDetector = BaseDetector
