"""
Drone Camera video source with telemetry timestamp synchronization.
"""

from typing import Tuple, Optional, Dict, Any
import time
import cv2
import numpy as np

from utils.logger import get_logger
from cameras.video_source import BaseVideoSource

logger = get_logger(__name__)


class DroneCamera(BaseVideoSource):
    """
    Drone video feed connecting via RTSP/UDP downlink with synchronized GPS metadata.
    """
    def __init__(
        self,
        source_id: str,
        stream_url: str,
        name: str = "Surveillance Drone Stream",
        telemetry_port: int = 14550,
    ):
        super().__init__(source_id=source_id, name=name)
        self.stream_url = stream_url
        self.telemetry_port = telemetry_port
        self.cap: Optional[cv2.VideoCapture] = None
        self._fps = 30.0
        self._resolution = (1920, 1080)
        self.current_telemetry: Dict[str, Any] = {
            "lat": 0.0,
            "lon": 0.0,
            "alt": 0.0,
            "heading": 0.0,
            "battery_pct": 100.0,
        }

    def open(self) -> bool:
        logger.info(f"Opening drone video stream '{self.stream_url}'...")
        self.cap = cv2.VideoCapture(self.stream_url)
        if not self.cap.isOpened():
            logger.warning(f"Could not connect to drone stream at {self.stream_url}")
            self.is_connected = False
            return False

        self._fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
        self._resolution = (w, h)
        self.is_connected = True
        logger.info(f"Drone camera active: {w}x{h} @ {self._fps:.1f} FPS")
        return True

    def update_telemetry(self, lat: float, lon: float, alt: float, heading: float, battery: float):
        """Updates internal telemetry state from MAVLink / drone controller."""
        self.current_telemetry = {
            "lat": lat,
            "lon": lon,
            "alt": alt,
            "heading": heading,
            "battery_pct": battery,
        }

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
        logger.info(f"Released drone camera '{self.name}'")

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def resolution(self) -> Tuple[int, int]:
        return self._resolution

    @property
    def source_type(self) -> str:
        return "drone"
