"""
USB Camera video source implementation.
"""

from typing import Tuple, Optional
import time
import cv2
import numpy as np

from utils.logger import get_logger
from cameras.video_source import BaseVideoSource

logger = get_logger(__name__)


class USBCamera(BaseVideoSource):
    """
    Direct USB video camera input via V4L2 (Linux) or DirectShow (Windows).
    """
    def __init__(
        self,
        source_id: str = "usb_cam_0",
        device_index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: float = 30.0,
    ):
        super().__init__(source_id=source_id, name=f"USB Camera {device_index}")
        self.device_index = device_index
        self._req_width = width
        self._req_height = height
        self._req_fps = fps
        self.cap: Optional[cv2.VideoCapture] = None
        self._actual_resolution = (width, height)
        self._actual_fps = fps

    def open(self) -> bool:
        logger.info(f"Connecting to USB camera index {self.device_index}...")
        self.cap = cv2.VideoCapture(self.device_index)
        if not self.cap.isOpened():
            logger.warning(f"Could not open USB camera index {self.device_index}")
            self.is_connected = False
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._req_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._req_height)
        self.cap.set(cv2.CAP_PROP_FPS, self._req_fps)

        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or self._req_width)
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or self._req_height)
        self._actual_resolution = (w, h)
        self._actual_fps = self.cap.get(cv2.CAP_PROP_FPS) or self._req_fps
        self.is_connected = True
        logger.info(f"Connected to USB camera {self.device_index}: {w}x{h} @ {self._actual_fps:.1f} FPS")
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
        logger.info(f"Released USB camera {self.device_index}")

    @property
    def fps(self) -> float:
        return self._actual_fps

    @property
    def resolution(self) -> Tuple[int, int]:
        return self._actual_resolution

    @property
    def source_type(self) -> str:
        return "usb"
