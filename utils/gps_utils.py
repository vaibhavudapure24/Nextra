"""
GPS and geo-spatial utilities for wildlife tracking:
- Haversine distance calculation
- Real-world ground distance calibration (Ground Sampling Distance / GSD)
- Geo-referencing image pixel coordinates to GPS coordinates
"""

import math
from typing import Tuple, Optional


def haversine_distance(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """
    Calculate the great-circle distance between two points on the Earth (in meters).
    """
    R = 6371000.0  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def pixel_to_gps(
    px: float,
    py: float,
    center_px: float,
    center_py: float,
    cam_lat: float,
    cam_lon: float,
    gsd_meters_per_pixel: float = 0.05,
    heading_deg: float = 0.0,
) -> Tuple[float, float]:
    """
    Converts image pixel offsets (relative to center) to approximate GPS coordinates
    given the camera's location, heading, and Ground Sampling Distance (GSD).

    Args:
        px, py: Pixel coordinate in the image
        center_px, center_py: Camera center pixel coordinate (e.g. 640, 360)
        cam_lat, cam_lon: Camera/Drone GPS coordinates
        gsd_meters_per_pixel: Scale in meters per pixel (default 5cm/px)
        heading_deg: Camera azimuth/heading in degrees (0 = North, 90 = East)

    Returns:
        (latitude, longitude)
    """
    dx_px = px - center_px
    dy_px = -(py - center_py)  # invert y so positive is North

    # Rotate by heading
    heading_rad = math.radians(heading_deg)
    dx_m = (dx_px * math.cos(heading_rad) + dy_px * math.sin(heading_rad)) * gsd_meters_per_pixel
    dy_m = (-dx_px * math.sin(heading_rad) + dy_px * math.cos(heading_rad)) * gsd_meters_per_pixel

    # Convert meter offsets to delta degrees
    # 1 deg latitude ~ 111,320 meters
    delta_lat = dy_m / 111320.0
    # 1 deg longitude ~ 111,320 * cos(lat) meters
    cos_lat = math.cos(math.radians(cam_lat))
    if abs(cos_lat) < 1e-6:
        cos_lat = 1e-6
    delta_lon = dx_m / (111320.0 * cos_lat)

    return cam_lat + delta_lat, cam_lon + delta_lon


def calculate_speed_mps(
    pos1: Tuple[float, float],
    pos2: Tuple[float, float],
    time_delta_sec: float,
    scale_m_per_pixel: float = 0.05,
    is_gps: bool = False,
) -> float:
    """
    Calculates animal speed in meters per second (m/s).
    """
    if time_delta_sec <= 0:
        return 0.0

    if is_gps:
        dist_m = haversine_distance(pos1[0], pos1[1], pos2[0], pos2[1])
    else:
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        dist_px = math.hypot(dx, dy)
        dist_m = dist_px * scale_m_per_pixel

    return dist_m / time_delta_sec
