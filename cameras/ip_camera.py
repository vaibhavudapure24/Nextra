"""
IP Camera video source implementation (HTTP/MJPEG/H.264 streams).
"""

from typing import Tuple, Optional
import time
import cv2
import numpy as np

from utils.logger import get_logger
from cameras.video_source import BaseVideoSource

logger = get_logger(__name__)


class IPCamera(BaseVideoSource):
    """
    IP Camera connecting over HTTP/MJPEG network endpoints.
    """
    def __init__(self, source_id: str, stream_url: str, name: str = "IP Camera"):
        super().__init__(source_id=source_id, name=name)
        self.stream_url = stream_url
        self.cap: Optional[cv2.VideoCapture] = None
        self._fps = 25.0
        self._resolution = (1280, 720)

    def open(self) -> bool:
        logger.info(f"Connecting to IP Camera stream '{self.stream_url}'...")
        self.cap = cv2.VideoCapture(self.stream_url)
        if not self.cap.isOpened():
            logger.warning(f"Could not open IP camera stream: {self.stream_url}")
            self.is_connected = False
            return False

        self._fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        self._resolution = (w, h)
        self.is_connected = True
        logger.info(f"Connected to IP camera '{self.name}': {w}x{h} @ {self._fps:.1f} FPS")
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        if self.cap is None or not self.is_connected:
            return False, None, time.time()
        ret, frame = self.cap.read()
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
        return "ip_camera"
