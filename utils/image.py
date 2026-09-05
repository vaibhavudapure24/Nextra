"""
Image processing utilities:
- Drawing bounding boxes, species tags, and tracking IDs
- Drawing motion trajectory trails
- Drawing virtual enclosure/fence polygons
- Low-light and night vision enhancement (CLAHE, gamma, denoising)
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

# Distinct color palette for species / track IDs
COLOR_PALETTE = [
    (46, 204, 113),  # Green
    (52, 152, 219),  # Blue
    (241, 196, 15),  # Yellow
    (230, 126, 34),  # Orange
    (155, 89, 182),  # Purple
    (26, 188, 156),  # Turquoise
    (231, 76, 60),   # Red
    (243, 156, 18),  # Amber
    (22, 160, 133),  # Dark cyan
    (142, 68, 173),  # Violet
]


def get_color_for_id(id_val: int) -> Tuple[int, int, int]:
    """Returns a deterministic RGB/BGR color tuple for a track or class ID."""
    return COLOR_PALETTE[abs(int(id_val)) % len(COLOR_PALETTE)]


def draw_bounding_box(
    image: np.ndarray,
    box: Tuple[float, float, float, float],
    label: str,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draws a styled bounding box with label background banner.
    """
    x1, y1, x2, y2 = map(int, box)
    h, w = image.shape[:2]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))

    # Draw rounded rectangle corner accents or solid box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    # Label text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    text_thickness = 1
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)

    # Label banner
    banner_y1 = max(0, y1 - th - baseline - 6)
    banner_y2 = y1
    banner_x2 = min(w, x1 + tw + 8)

    cv2.rectangle(image, (x1, banner_y1), (banner_x2, banner_y2), color, -1)
    # White or black text depending on color luminosity
    text_color = (0, 0, 0) if sum(color) > 380 else (255, 255, 255)
    cv2.putText(
        image,
        label,
        (x1 + 4, y1 - baseline - 2),
        font,
        font_scale,
        text_color,
        text_thickness,
        cv2.LINE_AA,
    )
    return image


def draw_trajectory(
    image: np.ndarray,
    trail: List[Tuple[float, float]],
    color: Tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draws a fading motion trajectory line connecting historical centroids.
    """
    if len(trail) < 2:
        return image

    num_points = len(trail)
    for i in range(1, num_points):
        p1 = (int(trail[i - 1][0]), int(trail[i - 1][1]))
        p2 = (int(trail[i][0]), int(trail[i][1]))
        # Gradual fade for older trail points
        alpha = (i / float(num_points))
        line_color = tuple(int(c * alpha) for c in color)
        line_thickness = max(1, int(thickness * alpha))
        cv2.line(image, p1, p2, line_color, line_thickness, cv2.LINE_AA)

    # Draw a circle on the current point
    curr = (int(trail[-1][0]), int(trail[-1][1]))
    cv2.circle(image, curr, 4, color, -1, cv2.LINE_AA)
    return image


def draw_fence_polygon(
    image: np.ndarray,
    polygon: List[List[int]],
    zone_name: str = "Virtual Enclosure",
    color: Tuple[int, int, int] = (0, 0, 255),
    alpha: float = 0.25,
) -> np.ndarray:
    """
    Draws a semi-transparent virtual fence polygon with border and label.
    """
    if len(polygon) < 3:
        return image

    pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
    overlay = image.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

    # Draw zone label near the first vertex
    lx, ly = polygon[0][0], polygon[0][1]
    cv2.putText(
        image,
        zone_name,
        (lx + 5, ly + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )
    return image


def enhance_night_vision(
    image: np.ndarray,
    clip_limit: float = 3.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
    gamma: float = 1.3,
    denoise: bool = False,
) -> np.ndarray:
    """
    Applies night-vision / low-light enhancement:
    1. Converts image to LAB color space.
    2. Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to Luminance channel.
    3. Converts back to BGR.
    4. Applies gamma correction for shadow recovery.
    5. Optional fast bilateral denoising.
    """
    if image is None or image.size == 0:
        return image

    # LAB CLAHE
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Gamma correction
    if abs(gamma - 1.0) > 0.05:
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        enhanced = cv2.LUT(enhanced, table)

    # Optional fast denoising
    if denoise:
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 5, 5, 7, 21)

    return enhanced
