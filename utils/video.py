"""
Video utility functions and helpers:
- FPS calculation
- Video writer initialization
- Frame queue management
- Threaded video capture for RTSP and live streams
"""

import time
import queue
import threading
from typing import Optional, Tuple
import cv2
import numpy as np
from utils.logger import get_logger

logger = get_logger(__name__)


class FPSCounter:
    """
    Computes rolling Average and Instantaneous FPS over an exponential window.
    """
    def __init__(self, alpha: float = 0.9):
        self.alpha = alpha
        self.last_time = time.perf_counter()
        self.fps = 0.0
        self.frame_count = 0
        self.start_time = time.perf_counter()

    def update(self) -> float:
        now = time.perf_counter()
        dt = now - self.last_time
        self.last_time = now
        self.frame_count += 1
        if dt > 0:
            current_fps = 1.0 / dt
            self.fps = current_fps if self.fps == 0.0 else (self.alpha * self.fps + (1 - self.alpha) * current_fps)
        return self.fps

    @property
    def average_fps(self) -> float:
        elapsed = time.perf_counter() - self.start_time
        return self.frame_count / elapsed if elapsed > 0 else 0.0


class VideoWriterWrapper:
    """
    Handles robust OpenCV VideoWriter setup and encoding.
    """
    def __init__(self, output_path: str, fps: float, width: int, height: int, codec: str = "mp4v"):
        self.output_path = output_path
        self.fps = fps
        self.width = width
        self.height = height
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not self.writer.isOpened():
            # Fallback to XVID / avi or avc1
            logger.warning(f"Failed to open VideoWriter with {codec}, attempting avc1/MJPG fallback")
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self.writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    def write(self, frame: np.ndarray) -> None:
        if self.writer and self.writer.isOpened():
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))
            self.writer.write(frame)

    def release(self) -> None:
        if self.writer:
            self.writer.release()
            self.writer = None
            logger.info(f"Closed video writer for {self.output_path}")


class ThreadedCameraReader:
    """
    Reads frames from an OpenCV VideoCapture in a background thread to prevent
    buffer latency and frame stalls during long AI inference cycles.
    """
    def __init__(self, src, max_queue_size: int = 4):
        self.cap = cv2.VideoCapture(src)
        self.q = queue.Queue(maxsize=max_queue_size)
        self.running = False
        self.thread: Optional[threading.Thread] = None

    def start(self):
        if self.running:
            return self
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        return self

    def _worker(self):
        while self.running:
            if not self.cap.isOpened():
                time.sleep(0.1)
                continue
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            # Drop older frames if full to maintain lowest possible live latency
            if self.q.full():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self, timeout: float = 1.0) -> Tuple[bool, Optional[np.ndarray]]:
        try:
            frame = self.q.get(timeout=timeout)
            return True, frame
        except queue.Empty:
            return False, None

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap.isOpened():
            self.cap.release()
