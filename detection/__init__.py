"""
Wildlife Animal Detection Module.
"""

from detection.postprocessing import DetectionResult, compute_iou, filter_by_confidence
from detection.detector import BaseDetector, get_detector
from detection.yolo11_detector import YOLO11Detector
from detection.yolov8_detector import YOLOv8Detector
from detection.inference import InferenceEngine

__all__ = [
    "DetectionResult",
    "compute_iou",
    "filter_by_confidence",
    "BaseDetector",
    "get_detector",
    "YOLO11Detector",
    "YOLOv8Detector",
    "InferenceEngine",
]
