"""
RTSP Camera video source with automatic reconnect and multithreaded buffering.
"""

from typing import Tuple, Optional
import time
import queue
import threading
import cv2
import numpy as np

from utils.logger import get_logger
from cameras.video_source import BaseVideoSource

logger = get_logger(__name__)


class RTSPCamera(BaseVideoSource):
    """
    RTSP IP streaming client with background frame reading and auto-reconnection.
    """
    def __init__(
        self,
        source_id: str,
        rtsp_url: str,
        name: str = "RTSP Stream",
        reconnect_delay_seconds: float = 5.0,
        max_queue_size: int = 3,
    ):
        super().__init__(source_id=source_id, name=name)
        self.rtsp_url = rtsp_url
        self.reconnect_delay = reconnect_delay_seconds
        self.max_queue_size = max_queue_size
        self._fps = 30.0
        self._resolution = (1920, 1080)

        self._frame_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None

    def open(self) -> bool:
        self._running = True
        self._worker_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._worker_thread.start()
        # Give connection up to 3 seconds to establish
        t_start = time.time()
        while time.time() - t_start < 3.0:
            if self.is_connected:
                return True
            time.sleep(0.1)
        return self.is_connected

    def _capture_loop(self):
        while self._running:
            # Mask credentials in logs for security
            masked_url = self.rtsp_url.split("@")[-1] if "@" in self.rtsp_url else self.rtsp_url
            logger.info(f"Connecting to RTSP source '{masked_url}'...")
            cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)

            if not cap.isOpened():
                logger.warning(f"RTSP stream offline. Retrying in {self.reconnect_delay}s...")
                self.is_connected = False
                time.sleep(self.reconnect_delay)
                continue

            self._fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
            self._resolution = (w, h)
            self.is_connected = True
            logger.info(f"RTSP connected: {masked_url} ({w}x{h} @ {self._fps:.1f} FPS)")

            consecutive_failures = 0
            while self._running and cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures > 30:
                        logger.warning("RTSP stream dropped frames. Reconnecting...")
                        break
                    time.sleep(0.01)
                    continue

                consecutive_failures = 0
                now = time.time()
                # Maintain freshest frame by dropping old ones
                if self._frame_queue.full():
                    try:
                        self._frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self._frame_queue.put((frame, now))

            cap.release()
            self.is_connected = False
            if self._running:
                time.sleep(self.reconnect_delay)

    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        try:
            frame, ts = self._frame_queue.get(timeout=0.5)
            self.frame_index += 1
            return True, frame, ts
        except queue.Empty:
            return False, None, time.time()

    def release(self) -> None:
        self._running = False
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
        self.is_connected = False
        logger.info(f"Closed RTSP stream '{self.name}'")

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def resolution(self) -> Tuple[int, int]:
        return self._resolution

    @property
    def source_type(self) -> str:
        return "rtsp"
