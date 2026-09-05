"""
Thermal Camera Interface and RGB + Thermal Fusion Pipeline.
Provides radiometric temperature calibration, color palette mapping, and multi-spectral fusion.
"""

from typing import Tuple, Optional
import cv2
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


class ThermalCameraInterface:
    """
    Interface for handling real thermal infrared streams.
    """
    def __init__(
        self,
        palette: str = "ironbow",
        temp_min_celsius: float = -10.0,
        temp_max_celsius: float = 50.0,
    ):
        self.palette = palette.lower()
        self.temp_min = temp_min_celsius
        self.temp_max = temp_max_celsius

    def apply_colormap(self, thermal_gray: np.ndarray) -> np.ndarray:
        """
        Applies radiometric false-color visualization to single-channel thermal frames.
        """
        if thermal_gray.ndim == 3 and thermal_gray.shape[2] == 3:
            thermal_gray = cv2.cvtColor(thermal_gray, cv2.COLOR_BGR2GRAY)

        # Normalize 8-bit
        if thermal_gray.dtype != np.uint8:
            norm_frame = cv2.normalize(thermal_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            norm_frame = thermal_gray

        if self.palette == "white_hot":
            return cv2.cvtColor(norm_frame, cv2.COLOR_GRAY2BGR)
        elif self.palette == "black_hot":
            inv = 255 - norm_frame
            return cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
        elif self.palette == "rainbow":
            return cv2.applyColorMap(norm_frame, cv2.COLORMAP_RAINBOW)
        else:
            # Default to ironbow / inferno
            return cv2.applyColorMap(norm_frame, cv2.COLORMAP_INFERNO)

    def estimate_temperature_map(self, raw_16bit: np.ndarray) -> np.ndarray:
        """
        Converts 16-bit radiometric raw pixel values to Celsius temperatures.
        """
        norm = raw_16bit.astype(np.float32) / 65535.0
        return self.temp_min + norm * (self.temp_max - self.temp_min)

    def fuse_rgb_thermal(
        self,
        rgb_frame: np.ndarray,
        thermal_frame: np.ndarray,
        alpha: float = 0.6,
    ) -> np.ndarray:
        """
        Performs multi-spectral fusion between aligned RGB and Thermal streams.
        """
        if thermal_frame.shape[:2] != rgb_frame.shape[:2]:
            thermal_frame = cv2.resize(thermal_frame, (rgb_frame.shape[1], rgb_frame.shape[0]))

        if thermal_frame.ndim == 2 or (thermal_frame.ndim == 3 and thermal_frame.shape[2] == 1):
            thermal_colored = self.apply_colormap(thermal_frame)
        else:
            thermal_colored = thermal_frame

        fused = cv2.addWeighted(rgb_frame, alpha, thermal_colored, 1.0 - alpha, 0.0)
        return fused
