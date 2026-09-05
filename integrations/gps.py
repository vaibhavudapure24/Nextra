"""
GPS Integration for wildlife tracking, camera geo-referencing, and drone telemetry.
Supports NMEA sentence parsing and coordinate transformations.
"""

from typing import Tuple, Optional, Dict, Any, List
import time
import datetime as dt
from utils.logger import get_logger
from utils.gps_utils import haversine_distance, pixel_to_gps

logger = get_logger(__name__)


class GPSCoordinate:
    """Represents a geo-spatial point with altitude and timestamp."""
    def __init__(
        self,
        latitude: float,
        longitude: float,
        altitude: float = 0.0,
        timestamp: Optional[float] = None,
        speed_mps: float = 0.0,
    ):
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.timestamp = timestamp or time.time()
        self.speed_mps = speed_mps

    def to_dict(self) -> Dict[str, Any]:
        return {
            "latitude": round(self.latitude, 6),
            "longitude": round(self.longitude, 6),
            "altitude": round(self.altitude, 1),
            "timestamp": self.timestamp,
            "speed_mps": round(self.speed_mps, 2),
        }


class GPSManager:
    """
    Manages GPS receivers and maintains location history for sensors, drones, and animals.
    """
    def __init__(self):
        self.history: Dict[str, List[GPSCoordinate]] = {}

    def parse_nmea_sentence(self, sentence: str) -> Optional[GPSCoordinate]:
        """
        Parses standard NMEA $GPGGA or $GPRMC strings.
        """
        parts = sentence.strip().split(",")
        if not parts or len(parts) < 6:
            return None

        msg_type = parts[0]
        try:
            if msg_type in ("$GPGGA", "$GNGGA"):
                # $GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*47
                lat_raw, lat_dir = parts[2], parts[3]
                lon_raw, lon_dir = parts[4], parts[5]
                alt = float(parts[9]) if len(parts) > 9 and parts[9] else 0.0

                lat = self._nmea_to_decimal(lat_raw, lat_dir)
                lon = self._nmea_to_decimal(lon_raw, lon_dir)
                return GPSCoordinate(latitude=lat, longitude=lon, altitude=alt)

            elif msg_type in ("$GPRMC", "$GNRMC"):
                # $GPRMC,123519,A,4807.038,N,01131.000,E,022.4,084.4,230394,003.1,W*6A
                status = parts[2]
                if status != "A":  # Data valid
                    return None
                lat_raw, lat_dir = parts[3], parts[4]
                lon_raw, lon_dir = parts[5], parts[6]
                speed_knots = float(parts[7]) if parts[7] else 0.0
                speed_mps = speed_knots * 0.514444

                lat = self._nmea_to_decimal(lat_raw, lat_dir)
                lon = self._nmea_to_decimal(lon_raw, lon_dir)
                return GPSCoordinate(latitude=lat, longitude=lon, speed_mps=speed_mps)

        except Exception as e:
            logger.debug(f"Error parsing NMEA sentence: {e}")
            return None

        return None

    @staticmethod
    def _nmea_to_decimal(raw: str, direction: str) -> float:
        if not raw:
            return 0.0
        # First 2 digits are degrees for lat, first 3 for lon
        dot_idx = raw.find(".")
        if dot_idx < 0:
            return 0.0
        deg_len = dot_idx - 2
        deg = float(raw[:deg_len])
        minutes = float(raw[deg_len:])
        decimal = deg + (minutes / 60.0)
        if direction in ("S", "W"):
            decimal = -decimal
        return decimal

    def record_location(self, entity_id: str, coord: GPSCoordinate) -> None:
        if entity_id not in self.history:
            self.history[entity_id] = []
        self.history[entity_id].append(coord)

    def get_path(self, entity_id: str) -> List[Dict[str, Any]]:
        return [c.to_dict() for c in self.history.get(entity_id, [])]
