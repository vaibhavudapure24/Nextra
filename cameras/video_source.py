"""
Unified VideoSource interface and VideoFile reader.
"""

from __future__ import annotations
import abc
import time
import os
from typing import Tuple, Optional, Any
import cv2
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


class BaseVideoSource(abc.ABC):
    """
    Abstract base class for all video capture sources.
    Decouples frame acquisition from detection and analytics.
    """
    def __init__(self, source_id: str, name: str):
        self.source_id = source_id
        self.name = name
        self.is_connected = False
        self.frame_index = 0

    def __enter__(self) -> BaseVideoSource:
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    def __iter__(self) -> BaseVideoSource:
        return self

    def __next__(self) -> Tuple[int, np.ndarray, float]:
        success, frame, timestamp = self.read()
        if not success or frame is None:
            raise StopIteration
        return self.frame_index, frame, timestamp

    @abc.abstractmethod
    def open(self) -> bool:
        """Opens the camera or video stream."""
        pass

    @abc.abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """
        Reads next frame.
        Returns: (success: bool, frame: np.ndarray or None, timestamp: float)
        """
        pass

    @abc.abstractmethod
    def release(self) -> None:
        """Releases underlying device/stream resources."""
        pass

    @property
    @abc.abstractmethod
    def fps(self) -> float:
        pass

    @property
    @abc.abstractmethod
    def resolution(self) -> Tuple[int, int]:
        pass

    @property
    @abc.abstractmethod
    def source_type(self) -> str:
        pass


class VideoFile(BaseVideoSource):
    """
    Reads frames from recorded video files with optional looping.
    """
    def __init__(self, source_id: str, file_path: str, loop: bool = True):
        super().__init__(source_id=source_id, name=os.path.basename(file_path))
        self.file_path = file_path
        self.loop = loop
        self.cap: Optional[cv2.VideoCapture] = None
        self._fps = 30.0
        self._resolution = (1280, 720)

    def open(self) -> bool:
        if not os.path.exists(self.file_path):
            logger.error(f"Video file not found: {self.file_path}")
            self.is_connected = False
            return False

        self.cap = cv2.VideoCapture(self.file_path)
        if not self.cap.isOpened():
            logger.error(f"Failed to open video file: {self.file_path}")
            self.is_connected = False
            return False

        self._fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        self._resolution = (w, h)
        self.is_connected = True
        logger.info(f"Opened VideoFile: '{self.file_path}' ({w}x{h} @ {self._fps:.1f} FPS)")
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        if self.cap is None or not self.is_connected:
            return False, None, time.time()

        ret, frame = self.cap.read()
        if not ret:
            if self.loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
            else:
                return False, None, time.time()

        if ret:
            self.frame_index += 1
            return True, frame, time.time()
        return False, None, time.time()

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_connected = False

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def resolution(self) -> Tuple[int, int]:
        return self._resolution

    @property
    def source_type(self) -> str:
        return "video_file"
