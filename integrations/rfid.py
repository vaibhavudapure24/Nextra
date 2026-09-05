"""
RFID Tag Reader Integration.
Associates microchip / ear-tag RFID transponder IDs with registered wildlife animal profiles.
"""

from typing import Optional, Dict, Any, Callable
import time
from utils.logger import get_logger

logger = get_logger(__name__)


class RFIDReaderInterface:
    """
    Interface for RFID tag scanners (e.g. at water troughs, feeding stations, or gates).
    """
    def __init__(self, port: Optional[str] = None, baudrate: int = 9600):
        self.port = port
        self.baudrate = baudrate
        self.is_connected = False
        self._tag_callback: Optional[Callable[[str, float], None]] = None

    def connect(self) -> bool:
        if not self.port:
            logger.info("No physical RFID port configured. Operating in simulated scan mode.")
            self.is_connected = True
            return True

        try:
            import serial
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            self.is_connected = True
            logger.info(f"Connected to physical RFID reader on {self.port}")
            return True
        except Exception as e:
            logger.warning(f"Could not connect to physical RFID port '{self.port}': {e}. Operating in virtual mode.")
            self.is_connected = True
            return True

    def register_callback(self, callback: Callable[[str, float], None]) -> None:
        self._tag_callback = callback

    def simulate_tag_scan(self, tag_uid: str) -> Dict[str, Any]:
        """
        Simulates scanning an RFID tag. Triggers callback and returns scan event.
        """
        ts = time.time()
        logger.info(f"RFID Scan Event: Tag UID '{tag_uid}' at timestamp {ts:.2f}")
        if self._tag_callback:
            self._tag_callback(tag_uid, ts)
        return {"tag_uid": tag_uid, "timestamp": ts, "status": "scanned"}
