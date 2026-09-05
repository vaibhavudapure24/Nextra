"""
SQLAlchemy ORM models for the AI Wildlife Animal Monitoring System.

Tables:
    animals            - Unique tracked animals (persistent identity, age/gender, reid embeddings)
    detections         - Detection events (species, bbox, confidence, camera, gps)
    tracking_events    - Trajectory, speed, distance travelled per frame/second
    behaviors          - Classified temporal behaviors (eating, sleeping, walking, aggression, etc.)
    health_records     - Observable health metrics (activity, stress, rest, distance, gait)
    anomalies          - Detected anomalies (fall, inactivity, running, aggression, escape)
    alerts             - High-priority operational alerts (INFO, LOW, MEDIUM, HIGH, CRITICAL)
    cameras            - Registered camera sources, RTSP streams, drones, thermal sensors
    video_sources      - Configured video input sources
    system_events      - Audit logs, system health, camera connect/disconnect
    reports            - Generated PDF and CSV wildlife census and health reports
    media_assets       - Snapshot images and video clips linked to events
"""

from __future__ import annotations
import enum
import datetime as dt
from typing import Optional, List

from sqlalchemy import (
    Column, Integer, Float, String, DateTime, Boolean, ForeignKey, Text, Enum as SAEnum, Index
)
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()


class AlertSeverity(str, enum.Enum):
    INFO = "info"
    LOW = "info"
    MEDIUM = "warning"
    HIGH = "warning"
    CRITICAL = "critical"
    info = "info"
    warning = "warning"
    critical = "critical"


class AlertType(str, enum.Enum):
    FENCE_ESCAPE = "fence_escape"
    INJURY = "injury_detected"
    STRESS = "stress_detected"
    AGGRESSION = "aggression_detected"
    PROLONGED_INACTIVITY = "no_movement"
    ABNORMAL_BEHAVIOR = "anomaly_generic"
    MEDICAL_EMERGENCY = "medical_emergency"
    CAMERA_FAILURE = "anomaly_generic"
    ANIMAL_FALL = "animal_fall"
    CONTINUOUS_RUNNING = "continuous_running"
    ISOLATION = "isolation"
    GENERIC = "anomaly_generic"
    no_movement = "no_movement"
    fence_escape = "fence_escape"
    stress_detected = "stress_detected"
    aggression_detected = "aggression_detected"
    injury_detected = "injury_detected"
    limping = "limping"
    continuous_running = "continuous_running"
    isolation = "isolation"
    medical_emergency = "medical_emergency"
    animal_fall = "animal_fall"
    anomaly_generic = "anomaly_generic"


class BehaviorType(str, enum.Enum):
    EATING = "eating"
    SLEEPING = "sleeping"
    WALKING = "walking"
    RUNNING = "running"
    RESTING = "resting"
    PLAYING = "playing"
    AGGRESSION = "aggression"
    STRESS = "stress"
    LIMPING = "limping"
    INJURY = "injury"
    SOCIAL_INTERACTION = "social_interaction"
    UNKNOWN = "unknown"


class CameraType(str, enum.Enum):
    USB = "usb"
    RTSP = "rtsp"
    IP = "ip"
    DRONE = "drone"
    VIDEO_FILE = "video_file"
    IMAGE_FOLDER = "image_folder"
    THERMAL = "thermal"


class MediaType(str, enum.Enum):
    IMAGE = "image"
    VIDEO = "video"


class ReportType(str, enum.Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CUSTOM = "custom"


class ReportFormat(str, enum.Enum):
    PDF = "pdf"
    CSV = "csv"


class Animal(Base):
    """A persistent animal entity with tracking and re-identification history."""
    __tablename__ = "animals"

    id = Column(Integer, primary_key=True, index=True)
    track_uid = Column(String(64), unique=True, nullable=False, index=True)
    species = Column(String(100), nullable=False, index=True)
    rfid_tag = Column(String(64), nullable=True, unique=True, index=True)
    first_seen = Column(DateTime, default=lambda: dt.datetime.now(dt.timezone.utc), index=True)
    last_seen = Column(DateTime, default=lambda: dt.datetime.now(dt.timezone.utc), index=True)
    estimated_age = Column(String(50), nullable=True)
    age_confidence = Column(Float, nullable=True)
    estimated_sex = Column(String(20), nullable=True)
    sex_confidence = Column(Float, nullable=True)
    reid_embedding = Column(Text, nullable=True)
    notes = Column(Text, nullable=True)
    active = Column(Boolean, default=True)

    # Relationships
    detections = relationship("Detection", back_populates="animal", cascade="all, delete-orphan")
    tracking_events = relationship("TrackingEvent", back_populates="animal", cascade="all, delete-orphan")
    behaviors = relationship("BehaviorRecord", back_populates="animal", cascade="all, delete-orphan")
    health_records = relationship("HealthRecord", back_populates="animal", cascade="all, delete-orphan")
    anomalies = relationship("AnomalyEvent", back_populates="animal", cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="animal", cascade="all, delete-orphan")


class Detection(Base):
    """Detection records from neural models."""
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)
    animal_id = Column(Integer, ForeignKey("animals.id"), nullable=True, index=True)
    detection_uid = Column(String(64), nullable=True, index=True)
    timestamp = Column(DateTime, default=lambda: dt.datetime.now(dt.timezone.utc), index=True)
    camera_id = Column(String(100), nullable=True, index=True)
    frame_number = Column(Integer, nullable=True)
    species = Column(String(100), nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    bbox_x1 = Column(Float, nullable=False)
    bbox_y1 = Column(Float, nullable=False)
    bbox_x2 = Column(Float, nullable=False)
    bbox_y2 = Column(Float, nullable=False)
    center_x = Column(Float, nullable=True)
    center_y = Column(Float, nullable=True)
    gps_lat = Column(Float, nullable=True)
    gps_lon = Column(Float, nullable=True)
    gps_alt = Column(Float, nullable=True)
    image_path = Column(String(500), nullable=True)

    animal = relationship("Animal", back_populates="detections")


class TrackingEvent(Base):
    """High-resolution tracking coordinates and kinematics."""
    __tablename__ = "tracking_events"

    id = Column(Integer, primary_key=True, index=True)
    animal_id = Column(Integer, ForeignKey("animals.id"), nullable=False, index=True)
    tracking_id = Column(Integer, nullable=False, index=True)
    timestamp = Column(DateTime, default=lambda: dt.datetime.now(dt.timezone.utc), index=True)
    camera_id = Column(String(100), nullable=True, index=True)
    speed_mps = Column(Float, default=0.0)
    distance_travelled_m = Column(Float, default=0.0)
    pos_x = Column(Float, nullable=False)
    pos_y = Column(Float, nullable=False)
    gps_lat = Column(Float, nullable=True)
    gps_lon = Column(Float, nullable=True)
    heading_deg = Column(Float, nullable=True)

    animal = relationship("Animal", back_populates="tracking_events")


class BehaviorRecord(Base):
    """Behavior classification history per animal."""
    __tablename__ = "behaviors"

    id = Column(Integer, primary_key=True, index=True)
    animal_id = Column(Integer, ForeignKey("animals.id"), nullable=False, index=True)
    timestamp = Column(DateTime, default=lambda: dt.datetime.now(dt.timezone.utc), index=True)
    behavior = Column(String(50), nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    duration_seconds = Column(Float, default=1.0)
    camera_id = Column(String(100), nullable=True)

    animal = relationship("Animal", back_populates="behaviors")


class HealthRecord(Base):
    """Computed health, gait, stress, and activity indices."""
    __tablename__ = "health_records"

    id = Column(Integer, primary_key=True, index=True)
    animal_id = Column(Integer, ForeignKey("animals.id"), nullable=False, index=True)
    timestamp = Column(DateTime, default=lambda: dt.datetime.now(dt.timezone.utc), index=True)
    activity_level = Column(Float, nullable=False)           # 0.0 - 1.0
    stress_index = Column(Float, nullable=False)             # 0.0 - 1.0
    rest_duration_min = Column(Float, default=0.0)
    sleeping_duration_min = Column(Float, default=0.0)
    feeding_duration_min = Column(Float, default=0.0)
    walking_distance_m = Column(Float, default=0.0)
    daily_movement_m = Column(Float, default=0.0)
    injury_risk_score = Column(Float, default=0.0)
    diagnostics = Column(Text, nullable=True)

    animal = relationship("Animal", back_populates="health_records")


class AnomalyEvent(Base):
    """Detected anomaly occurrences."""
    __tablename__ = "anomalies"

    id = Column(Integer, primary_key=True, index=True)
    animal_id = Column(Integer, ForeignKey("animals.id"), nullable=True, index=True)
    timestamp = Column(DateTime, default=lambda: dt.datetime.now(dt.timezone.utc), index=True)
    anomaly_type = Column(String(100), nullable=False, index=True)
    anomaly_score = Column(Float, nullable=False)
    severity = Column(String(20), nullable=False, default="MEDIUM")
    confidence = Column(Float, nullable=False, default=0.8)
    camera_id = Column(String(100), nullable=True)
    details = Column(Text, nullable=True)

    animal = relationship("Animal", back_populates="anomalies")


class Alert(Base):
    """System-wide operational alerts across all severity tiers."""
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    animal_id = Column(Integer, ForeignKey("animals.id"), nullable=True, index=True)
    alert_type = Column(String(50), nullable=False, index=True)
    severity = Column(String(20), nullable=False, default="MEDIUM", index=True)
    message = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=lambda: dt.datetime.now(dt.timezone.utc), index=True)
    camera_id = Column(String(100), nullable=True, index=True)
    zone_name = Column(String(100), nullable=True)
    gps_lat = Column(Float, nullable=True)
    gps_lon = Column(Float, nullable=True)
    snapshot_path = Column(String(500), nullable=True)
    video_path = Column(String(500), nullable=True)
    acknowledged = Column(Boolean, default=False, index=True)
    notified_email = Column(Boolean, default=False)
    notified_telegram = Column(Boolean, default=False)
    notified_sms = Column(Boolean, default=False)

    animal = relationship("Animal", back_populates="alerts")


class Camera(Base):
    """Registered camera streams."""
    __tablename__ = "cameras"

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(String(100), unique=True, nullable=False, index=True)
    name = Column(String(150), nullable=False)
    camera_type = Column(SAEnum(CameraType), nullable=False, default=CameraType.RTSP)
    url = Column(String(500), nullable=False)
    fps = Column(Float, default=30.0)
    resolution_w = Column(Integer, default=1280)
    resolution_h = Column(Integer, default=720)
    gps_lat = Column(Float, nullable=True)
    gps_lon = Column(Float, nullable=True)
    gps_alt = Column(Float, nullable=True)
    is_active = Column(Boolean, default=True)
    night_vision_enabled = Column(Boolean, default=False)
    last_ping = Column(DateTime, nullable=True)


class VideoSourceModel(Base):
    """Video input configurations."""
    __tablename__ = "video_sources"

    id = Column(Integer, primary_key=True, index=True)
    source_name = Column(String(100), unique=True, nullable=False)
    source_type = Column(String(50), nullable=False)
    uri = Column(String(500), nullable=False)
    enabled = Column(Boolean, default=True)
    description = Column(Text, nullable=True)


class SystemEvent(Base):
    """System-level diagnostic and operational event log."""
    __tablename__ = "system_events"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=lambda: dt.datetime.now(dt.timezone.utc), index=True)
    event_type = Column(String(100), nullable=False, index=True)
    component = Column(String(100), nullable=False, index=True)
    message = Column(Text, nullable=False)
    level = Column(String(20), default="INFO")
    meta_json = Column(Text, nullable=True)


class Report(Base):
    """Generated census, daily, weekly, and health reports."""
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    report_type = Column(String(50), nullable=False, index=True)
    report_format = Column(String(20), nullable=False)
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    file_path = Column(String(500), nullable=False)
    summary = Column(Text, nullable=True)
    generated_at = Column(DateTime, default=lambda: dt.datetime.now(dt.timezone.utc), index=True)


class MediaAsset(Base):
    """Captured snapshot images and recorded clips."""
    __tablename__ = "media_assets"

    id = Column(Integer, primary_key=True, index=True)
    media_type = Column(SAEnum(MediaType), nullable=False)
    file_path = Column(String(500), nullable=False)
    timestamp = Column(DateTime, default=lambda: dt.datetime.now(dt.timezone.utc), index=True)
    camera_id = Column(String(100), nullable=True)
    animal_id = Column(Integer, ForeignKey("animals.id"), nullable=True)
