"""
Image Folder video source (simulates video stream from a sequence of images or camera trap SD card).
"""

from typing import Tuple, Optional, List
import time
import os
import cv2
import numpy as np

from utils.logger import get_logger
from cameras.video_source import BaseVideoSource

logger = get_logger(__name__)


class ImageFolder(BaseVideoSource):
    """
    Sequences through a directory of image files at a specified FPS.
    """
    def __init__(
        self,
        source_id: str,
        folder_path: str,
        fps: float = 5.0,
        loop: bool = True,
    ):
        super().__init__(source_id=source_id, name=f"ImageFolder ({os.path.basename(folder_path)})")
        self.folder_path = folder_path
        self._fps = fps
        self.loop = loop
        self.image_files: List[str] = []
        self.current_idx = 0
        self._resolution = (1280, 720)

    def open(self) -> bool:
        if not os.path.exists(self.folder_path):
            logger.error(f"Image folder does not exist: {self.folder_path}")
            self.is_connected = False
            return False

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        files = [
            os.path.join(self.folder_path, f)
            for f in sorted(os.listdir(self.folder_path))
            if os.path.splitext(f)[1].lower() in valid_exts
        ]

        if not files:
            logger.warning(f"No valid image files found in {self.folder_path}")
            self.is_connected = False
            return False

        self.image_files = files
        self.current_idx = 0
        self.is_connected = True

        # Read first image to determine resolution
        first_img = cv2.imread(self.image_files[0])
        if first_img is not None:
            self._resolution = (first_img.shape[1], first_img.shape[0])

        logger.info(f"Loaded {len(files)} images from '{self.folder_path}'")
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        if not self.is_connected or not self.image_files:
            return False, None, time.time()

        if self.current_idx >= len(self.image_files):
            if self.loop:
                self.current_idx = 0
            else:
                return False, None, time.time()

        img_path = self.image_files[self.current_idx]
        self.current_idx += 1
        frame = cv2.imread(img_path)

        if frame is None:
            return False, None, time.time()

        self.frame_index += 1
        return True, frame, time.time()

    def release(self) -> None:
        self.image_files.clear()
        self.is_connected = False

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def resolution(self) -> Tuple[int, int]:
        return self._resolution

    @property
    def source_type(self) -> str:
        return "image_folder"
