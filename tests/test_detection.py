import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import numpy as np

from detection.postprocessing import DetectionResult, compute_iou, filter_by_confidence
from detection.preprocessing import letterbox, normalize_image
from detection.detector import get_detector


def test_detection_result_creation():
    det = DetectionResult.create(
        class_id=1,
        species="zebra",
        confidence=0.92,
        bbox=(100.0, 200.0, 300.0, 500.0),
    )
    assert det.species == "zebra"
    assert det.confidence == 0.92
    assert det.width == 200.0
    assert det.height == 300.0
    assert det.center_x == 200.0
    assert det.center_y == 350.0
    assert det.area == 60000.0


def test_compute_iou():
    box1 = (0.0, 0.0, 10.0, 10.0)
    box2 = (5.0, 0.0, 15.0, 10.0)
    iou = compute_iou(box1, box2)
    # Intersection = 5 * 10 = 50. Union = 100 + 100 - 50 = 150. IoU = 50 / 150 = 0.3333
    assert pytest.approx(iou, 0.01) == 0.333


def test_filter_by_confidence():
    d1 = DetectionResult.create(class_id=0, species="elephant", confidence=0.85, bbox=(0, 0, 10, 10))
    d2 = DetectionResult.create(class_id=0, species="elephant", confidence=0.35, bbox=(0, 0, 10, 10))
    filtered = filter_by_confidence([d1, d2], min_confidence=0.50)
    assert len(filtered) == 1
    assert filtered[0].confidence == 0.85


def test_preprocessing_letterbox():
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    resized, ratio, pad = letterbox(dummy, new_shape=(640, 640), auto=False)
    assert resized.shape == (640, 640, 3)
    norm = normalize_image(resized)
    assert norm.shape == (3, 640, 640)
    assert norm.dtype == np.float32


def test_detector_inference_smoke():
    detector = get_detector(confidence_threshold=0.30, device="cpu")
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    detections = detector.detect(dummy)
    assert isinstance(detections, list)
