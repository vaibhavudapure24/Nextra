"""
Unified video source abstraction.

Supports:
    - USB camera            (VideoSourceType.USB)
    - RTSP camera           (VideoSourceType.RTSP)
    - Generic IP camera     (VideoSourceType.IP_CAMERA)
    - Drone camera (RTSP)   (VideoSourceType.DRONE)
    - Recorded video file   (VideoSourceType.VIDEO_FILE)
    - Folder of images      (VideoSourceType.IMAGE_FOLDER)

All sources implement the same iterator-style interface so the rest of the
pipeline (detection/tracking/behavior/etc.) never needs to know where the
frames came from:

    source = get_video_source(cfg)
    with source:
        for frame_id, frame, timestamp in source:
            ...

Network sources (RTSP/IP/drone) auto-reconnect on read failure.
"""

from __future__ import annotations

import time
import glob
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


class VideoSourceType(str, Enum):
    USB = "usb"
    RTSP = "rtsp"
    IP_CAMERA = "ip_camera"
    DRONE = "drone"
    VIDEO_FILE = "video_file"
    IMAGE_FOLDER = "image_folder"


class BaseVideoSource(ABC):
    """Common interface for every video source type."""

    source_type: VideoSourceType

    def __enter__(self) -> "BaseVideoSource":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    @abstractmethod
    def open(self) -> None:
        ...

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Return (success, frame_bgr)."""
        ...

    @abstractmethod
    def release(self) -> None:
        ...

    @property
    def fps(self) -> float:
        return 30.0

    def __iter__(self) -> Iterator[Tuple[int, np.ndarray, float]]:
        frame_id = 0
        while True:
            ok, frame = self.read()
            if not ok or frame is None:
                if not self._should_continue_after_failure():
                    break
                continue
            yield frame_id, frame, time.time()
            frame_id += 1

    def _should_continue_after_failure(self) -> bool:
        """Override in subclasses to control retry/reconnect vs. stop-iteration behavior."""
        return False


class _OpenCVCaptureSource(BaseVideoSource):
    """Shared implementation for anything backed by cv2.VideoCapture."""

    def __init__(self, uri: str | int, reconnect_delay_seconds: float = 5.0,
                 width: Optional[int] = None, height: Optional[int] = None,
                 target_fps: Optional[int] = None, max_consecutive_failures: int = 150):
        self.uri = uri
        self.reconnect_delay_seconds = reconnect_delay_seconds
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.max_consecutive_failures = max_consecutive_failures
        self.cap: Optional[cv2.VideoCapture] = None
        self._consecutive_failures = 0

    def open(self) -> None:
        logger.info(f"Opening video capture: {self.uri}")
        self.cap = cv2.VideoCapture(self.uri)
        if self.width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.target_fps:
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        if not self.cap.isOpened():
            logger.error(
                f"Failed to open video source: {self.uri}. "
                f"If this is a USB camera, check the device index (try 0, 1, 2...), "
                f"that no other app is using it, and that your user has camera permissions "
                f"(on Linux: `groups $USER` should include 'video', or run `ls -l /dev/video*`)."
            )

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.cap is None or not self.cap.isOpened():
            return False, None
        ok, frame = self.cap.read()
        if ok:
            self._consecutive_failures = 0
        return ok, frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            logger.info(f"Released video source: {self.uri}")

    @property
    def fps(self) -> float:
        if self.cap is not None:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            return fps if fps and fps > 0 else 30.0
        return 30.0


class USBCameraSource(_OpenCVCaptureSource):
    source_type = VideoSourceType.USB

    def _should_continue_after_failure(self) -> bool:
        # A single dropped USB frame is usually transient; keep going for a while.
        # But if EVERY read keeps failing, the camera likely never opened at all
        # (wrong index / permissions / in use) -- give up loudly instead of
        # spinning forever with no feedback.
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.max_consecutive_failures:
            logger.error(
                f"No frames received from USB camera '{self.uri}' after "
                f"{self._consecutive_failures} consecutive attempts. Giving up. "
                f"This almost always means the camera never opened successfully — "
                f"see the 'Failed to open video source' message above, or try "
                f"`ls /dev/video*` to find the correct device index."
            )
            return False
        time.sleep(0.05)
        return True


class _NetworkStreamSource(_OpenCVCaptureSource):
    """Common auto-reconnect behavior for RTSP / IP camera / drone streams."""

    def _should_continue_after_failure(self) -> bool:
        logger.warning(
            f"Lost connection to network stream '{self.uri}'. "
            f"Reconnecting in {self.reconnect_delay_seconds}s..."
        )
        self.release()
        time.sleep(self.reconnect_delay_seconds)
        self.open()
        return True


class RTSPSource(_NetworkStreamSource):
    source_type = VideoSourceType.RTSP


class IPCameraSource(_NetworkStreamSource):
    source_type = VideoSourceType.IP_CAMERA


class DroneSource(_NetworkStreamSource):
    """
    Drone camera source. In practice this is almost always an RTSP (or RTMP)
    stream published by the drone's onboard transmitter / ground station
    (e.g. DJI, ArduPilot companion computers). If your drone SDK exposes a
    different transport (e.g. a proprietary UDP video protocol), implement
    a dedicated subclass of BaseVideoSource that decodes it into BGR frames
    and reuses the rest of the pipeline unchanged.
    """
    source_type = VideoSourceType.DRONE


class VideoFileSource(BaseVideoSource):
    source_type = VideoSourceType.VIDEO_FILE

    def __init__(self, path: str, loop: bool = True):
        self.path = path
        self.loop = loop
        self.cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        if not Path(self.path).exists():
            logger.error(f"Video file not found: {self.path}")
        self.cap = cv2.VideoCapture(self.path)
        logger.info(f"Opened video file: {self.path}")

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        ok, frame = self.cap.read()
        if not ok and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        return ok, frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()

    @property
    def fps(self) -> float:
        if self.cap is not None:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            return fps if fps and fps > 0 else 30.0
        return 30.0


class ImageFolderSource(BaseVideoSource):
    """Treats a directory of still images as a low-frame-rate video stream."""
    source_type = VideoSourceType.IMAGE_FOLDER

    SUPPORTED_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp")

    def __init__(self, folder: str, target_fps: int = 5):
        self.folder = folder
        self.target_fps = max(target_fps, 1)
        self.frame_delay = 1.0 / self.target_fps
        self.image_paths: list[str] = []
        self._index = 0

    def open(self) -> None:
        paths: list[str] = []
        for ext in self.SUPPORTED_EXTENSIONS:
            paths.extend(glob.glob(str(Path(self.folder) / ext)))
        self.image_paths = sorted(paths)
        self._index = 0
        logger.info(f"Loaded {len(self.image_paths)} images from {self.folder}")
        if not self.image_paths:
            logger.warning(f"No images found in {self.folder}")

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._index >= len(self.image_paths):
            return False, None
        frame = cv2.imread(self.image_paths[self._index])
        self._index += 1
        time.sleep(self.frame_delay)
        return frame is not None, frame

    def release(self) -> None:
        self._index = 0

    @property
    def fps(self) -> float:
        return float(self.target_fps)


def get_video_source(cfg, source_type: Optional[str] = None) -> BaseVideoSource:
    """
    Factory that builds the right BaseVideoSource subclass from the loaded
    config (utils.config_loader.load_config()).

    Args:
        cfg: the loaded ConfigNode (cfg.video_sources.*)
        source_type: optional override, one of the VideoSourceType values.
                     Falls back to cfg.video_sources.default.
    """
    vs_cfg = cfg.video_sources
    stype = VideoSourceType(source_type or vs_cfg.default)

    if stype == VideoSourceType.USB:
        c = vs_cfg.usb
        return USBCameraSource(c.device_index, width=c.width, height=c.height, target_fps=c.fps)

    if stype == VideoSourceType.RTSP:
        c = vs_cfg.rtsp
        return RTSPSource(c.url, reconnect_delay_seconds=c.reconnect_delay_seconds)

    if stype == VideoSourceType.IP_CAMERA:
        c = vs_cfg.ip_camera
        return IPCameraSource(c.url)

    if stype == VideoSourceType.DRONE:
        c = vs_cfg.drone
        return DroneSource(c.url, reconnect_delay_seconds=c.reconnect_delay_seconds)

    if stype == VideoSourceType.VIDEO_FILE:
        c = vs_cfg.video_file
        return VideoFileSource(c.path, loop=c.loop)

    if stype == VideoSourceType.IMAGE_FOLDER:
        c = vs_cfg.image_folder
        return ImageFolderSource(c.path, target_fps=c.fps)

    raise ValueError(f"Unsupported video source type: {stype}")
