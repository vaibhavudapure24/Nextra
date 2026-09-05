"""
Detection inference engine for batch processing and stream pipeline execution.
"""

from typing import List, Generator, Tuple, Optional
import numpy as np

from utils.logger import get_logger
from detection.detector import BaseDetector, get_detector
from detection.postprocessing import DetectionResult

logger = get_logger(__name__)


class InferenceEngine:
    """
    Manages detection inference on frames, video files, and camera streams.
    """
    def __init__(self, detector: Optional[BaseDetector] = None):
        self.detector = detector or get_detector()

    def process_frame(
        self,
        frame: np.ndarray,
        confidence_threshold: Optional[float] = None,
        frame_number: int = 0,
        draw: bool = False,
    ) -> Tuple[List[DetectionResult], Optional[np.ndarray]]:
        """
        Executes detection on a single frame and optionally returns the annotated frame.
        """
        detections = self.detector.detect(
            frame=frame,
            confidence_threshold=confidence_threshold,
            frame_number=frame_number,
        )
        annotated = self.detector.draw(frame, detections) if draw else None
        return detections, annotated

    def process_batch(
        self,
        frames: List[np.ndarray],
        confidence_threshold: Optional[float] = None,
    ) -> List[List[DetectionResult]]:
        """
        Processes a batch of frames sequentially or via batched inference.
        """
        results = []
        for idx, frame in enumerate(frames):
            dets = self.detector.detect(
                frame=frame,
                confidence_threshold=confidence_threshold,
                frame_number=idx,
            )
            results.append(dets)
        return results
