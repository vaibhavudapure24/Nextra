"""
Hardware and Sensor Integrations Module.
"""

from integrations.gps import GPSCoordinate, GPSManager
from integrations.thermal import ThermalCameraInterface
from integrations.rfid import RFIDReaderInterface
from integrations.drone import DroneTelemetryAdapter

__all__ = [
    "GPSCoordinate",
    "GPSManager",
    "ThermalCameraInterface",
    "RFIDReaderInterface",
    "DroneTelemetryAdapter",
]
