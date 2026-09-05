"""
YOLOv8 Detector compatibility implementation for Wildlife Monitoring System.
"""

import os
from typing import List, Optional
import numpy as np

from utils.logger import get_logger
from utils.config_loader import resolve_path
from detection.detector import BaseDetector
from detection.postprocessing import DetectionResult

logger = get_logger(__name__)


class YOLOv8Detector(BaseDetector):
    """
    Detector implementation supporting YOLOv8 checkpoints.
    """
    def _load_model(self) -> None:
        from ultralytics import YOLO

        resolved = str(resolve_path(self.model_path))
        if not os.path.exists(resolved) and not resolved.endswith(".pt"):
            resolved = "yolov8n.pt"
            logger.info(f"Custom model '{self.model_path}' not found locally; auto-loading base YOLOv8 weights: yolov8n.pt")
        elif not os.path.exists(resolved) and resolved.endswith(".pt") and not os.path.isabs(resolved):
            resolved = os.path.basename(resolved) if "yolov8" in resolved else "yolov8n.pt"

        logger.info(f"Initializing YOLOv8 detector from '{resolved}' on {self.torch_device}")
        try:
            self.model = YOLO(resolved)
            self.model.to(str(self.torch_device))
        except Exception as e:
            logger.warning(f"Error loading YOLOv8 model on {self.torch_device}: {e}. Retrying on CPU.")
            self.model = YOLO("yolov8n.pt")
            self.model.to("cpu")

    def detect(
        self,
        frame: np.ndarray,
        confidence_threshold: Optional[float] = None,
        frame_number: int = 0,
        timestamp: Optional[float] = None,
    ) -> List[DetectionResult]:
        if self.model is None or frame is None or frame.size == 0:
            return []

        conf = confidence_threshold if confidence_threshold is not None else self.confidence_threshold
        is_cuda = str(self.torch_device).startswith("cuda")

        try:
            results = self.model.predict(
                source=frame,
                conf=conf,
                iou=self.iou_threshold,
                device=str(self.torch_device),
                half=is_cuda,
                verbose=False,
            )
        except Exception as exc:
            logger.error(f"YOLOv8 inference error: {exc}")
            return []

        detections: List[DetectionResult] = []
        if not results:
            return detections

        res = results[0]
        names = res.names or {}

        if res.boxes is not None:
            for box in res.boxes:
                cls_id = int(box.cls.item())
                confidence = float(box.conf.item())
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                fallback_name = names.get(cls_id, f"animal_{cls_id}")
                species_name = self.get_species_name(cls_id, fallback_name=fallback_name)

                det = DetectionResult.create(
                    class_id=cls_id,
                    species=species_name,
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                    frame_number=frame_number,
                    timestamp=timestamp,
                )
                detections.append(det)

        return detections
