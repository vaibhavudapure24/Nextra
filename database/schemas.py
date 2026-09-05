"""
Pydantic schemas for data validation and API serialization.
"""

from __future__ import annotations
import datetime as dt
from typing import Optional, List, Dict, Any, Tuple
from pydantic import BaseModel, Field
from database.models import AlertSeverity, AlertType, BehaviorType, CameraType, ReportType, ReportFormat


class AnimalBase(BaseModel):
    track_uid: str
    species: str
    rfid_tag: Optional[str] = None
    estimated_age: Optional[str] = None
    age_confidence: Optional[float] = None
    estimated_sex: Optional[str] = None
    sex_confidence: Optional[float] = None
    notes: Optional[str] = None
    active: bool = True


class AnimalCreate(AnimalBase):
    pass


class AnimalResponse(AnimalBase):
    id: int
    first_seen: dt.datetime
    last_seen: dt.datetime

    class Config:
        from_attributes = True


class DetectionBase(BaseModel):
    species: str
    confidence: float
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    center_x: Optional[float] = None
    center_y: Optional[float] = None
    camera_id: Optional[str] = None
    frame_number: Optional[int] = None
    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None
    gps_alt: Optional[float] = None
    image_path: Optional[str] = None


class DetectionCreate(DetectionBase):
    animal_id: Optional[int] = None
    detection_uid: Optional[str] = None


class DetectionResponse(DetectionBase):
    id: int
    animal_id: Optional[int] = None
    detection_uid: Optional[str] = None
    timestamp: dt.datetime

    class Config:
        from_attributes = True


class TrackingEventCreate(BaseModel):
    animal_id: int
    tracking_id: int
    camera_id: Optional[str] = None
    speed_mps: float = 0.0
    distance_travelled_m: float = 0.0
    pos_x: float
    pos_y: float
    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None
    heading_deg: Optional[float] = None


class TrackingEventResponse(TrackingEventCreate):
    id: int
    timestamp: dt.datetime

    class Config:
        from_attributes = True


class BehaviorRecordCreate(BaseModel):
    animal_id: int
    behavior: str
    confidence: float
    duration_seconds: float = 1.0
    camera_id: Optional[str] = None


class BehaviorRecordResponse(BehaviorRecordCreate):
    id: int
    timestamp: dt.datetime

    class Config:
        from_attributes = True


class HealthRecordCreate(BaseModel):
    animal_id: int
    activity_level: float
    stress_index: float
    rest_duration_min: float = 0.0
    sleeping_duration_min: float = 0.0
    feeding_duration_min: float = 0.0
    walking_distance_m: float = 0.0
    daily_movement_m: float = 0.0
    injury_risk_score: float = 0.0
    diagnostics: Optional[str] = None


class HealthRecordResponse(HealthRecordCreate):
    id: int
    timestamp: dt.datetime

    class Config:
        from_attributes = True


class AnomalyEventCreate(BaseModel):
    animal_id: Optional[int] = None
    anomaly_type: str
    anomaly_score: float
    severity: AlertSeverity = AlertSeverity.MEDIUM
    confidence: float = 0.8
    camera_id: Optional[str] = None
    details: Optional[str] = None


class AnomalyEventResponse(AnomalyEventCreate):
    id: int
    timestamp: dt.datetime

    class Config:
        from_attributes = True


class AlertCreate(BaseModel):
    animal_id: Optional[int] = None
    alert_type: AlertType
    severity: AlertSeverity = AlertSeverity.MEDIUM
    message: str
    camera_id: Optional[str] = None
    zone_name: Optional[str] = None
    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None
    snapshot_path: Optional[str] = None
    video_path: Optional[str] = None


class AlertResponse(AlertCreate):
    id: int
    timestamp: dt.datetime
    acknowledged: bool
    notified_email: bool
    notified_telegram: bool
    notified_sms: bool

    class Config:
        from_attributes = True


class CameraCreate(BaseModel):
    camera_id: str
    name: str
    camera_type: CameraType = CameraType.RTSP
    url: str
    fps: float = 30.0
    resolution_w: int = 1280
    resolution_h: int = 720
    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None
    gps_alt: Optional[float] = None
    night_vision_enabled: bool = False


class CameraResponse(CameraCreate):
    id: int
    is_active: bool
    last_ping: Optional[dt.datetime] = None

    class Config:
        from_attributes = True


class ReportResponse(BaseModel):
    id: int
    report_type: ReportType
    report_format: ReportFormat
    period_start: dt.datetime
    period_end: dt.datetime
    file_path: str
    summary: Optional[str] = None
    generated_at: dt.datetime

    class Config:
        from_attributes = True


# Request DTOs
class DetectRequest(BaseModel):
    camera_id: Optional[str] = "cam_01"
    confidence_threshold: Optional[float] = 0.40
    device: Optional[str] = "auto"
    image_base64: Optional[str] = None


class TrackRequest(BaseModel):
    tracker_type: Optional[str] = "bytetrack"
    detections: List[Dict[str, Any]]
    camera_id: Optional[str] = "cam_01"


class BehaviorInferenceRequest(BaseModel):
    animal_id: int
    temporal_features: List[List[float]]
    model_type: Optional[str] = "lstm"


class AnomalyInferenceRequest(BaseModel):
    features: List[float]
    method: Optional[str] = "isolation_forest"
    animal_id: Optional[int] = None
