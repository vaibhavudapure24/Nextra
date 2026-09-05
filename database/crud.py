"""
CRUD helper functions for database operations in the AI Wildlife Monitoring System.
"""

from __future__ import annotations
import datetime as dt
from typing import Optional, Sequence, List, Dict, Any

from sqlalchemy import func, desc
from sqlalchemy.orm import Session

from database.models import (
    Animal, Detection, TrackingEvent, BehaviorRecord, HealthRecord,
    AnomalyEvent, Alert, Camera, VideoSourceModel, SystemEvent, Report, MediaAsset,
    AlertType, AlertSeverity, ReportType, ReportFormat
)
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Animals
# ---------------------------------------------------------------------------

def get_or_create_animal(db: Session, track_uid: str, species: str) -> Animal:
    """Retrieves an animal by unique tracking ID or creates a new record."""
    animal = db.query(Animal).filter(Animal.track_uid == track_uid).first()
    now = dt.datetime.now(dt.timezone.utc)
    if animal is None:
        animal = Animal(
            track_uid=track_uid,
            species=species,
            first_seen=now,
            last_seen=now,
            active=True,
        )
        db.add(animal)
        db.commit()
        db.refresh(animal)
        logger.info(f"Registered new animal: {track_uid} ({species})")
    else:
        animal.last_seen = now
        animal.active = True
        db.commit()
        db.refresh(animal)
    return animal


def get_animal_by_id(db: Session, animal_id: int) -> Optional[Animal]:
    return db.query(Animal).filter(Animal.id == animal_id).first()


def get_animal_by_uid(db: Session, track_uid: str) -> Optional[Animal]:
    return db.query(Animal).filter(Animal.track_uid == track_uid).first()


def list_animals(
    db: Session, species: Optional[str] = None, active_only: bool = False, limit: int = 100
) -> Sequence[Animal]:
    q = db.query(Animal)
    if species:
        q = q.filter(Animal.species == species)
    if active_only:
        q = q.filter(Animal.active.is_(True))
    return q.order_by(desc(Animal.last_seen)).limit(limit).all()


# ---------------------------------------------------------------------------
# Detections
# ---------------------------------------------------------------------------

def log_detection(
    db: Session,
    species: str,
    confidence: float,
    bbox: tuple[float, float, float, float],
    animal_id: Optional[int] = None,
    detection_uid: Optional[str] = None,
    camera_id: Optional[str] = None,
    frame_number: Optional[int] = None,
    gps: Optional[tuple[float, float, float]] = None,
    image_path: Optional[str] = None,
) -> Detection:
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    det = Detection(
        animal_id=animal_id,
        detection_uid=detection_uid,
        species=species,
        confidence=confidence,
        bbox_x1=bbox[0],
        bbox_y1=bbox[1],
        bbox_x2=bbox[2],
        bbox_y2=bbox[3],
        center_x=cx,
        center_y=cy,
        camera_id=camera_id,
        frame_number=frame_number,
        gps_lat=gps[0] if gps else None,
        gps_lon=gps[1] if gps else None,
        gps_alt=gps[2] if gps and len(gps) > 2 else None,
        image_path=image_path,
    )
    db.add(det)
    db.commit()
    db.refresh(det)
    return det


def get_detection_history(
    db: Session,
    animal_id: Optional[int] = None,
    species: Optional[str] = None,
    camera_id: Optional[str] = None,
    limit: int = 200,
) -> Sequence[Detection]:
    q = db.query(Detection)
    if animal_id is not None:
        q = q.filter(Detection.animal_id == animal_id)
    if species:
        q = q.filter(Detection.species == species)
    if camera_id:
        q = q.filter(Detection.camera_id == camera_id)
    return q.order_by(desc(Detection.timestamp)).limit(limit).all()


# ---------------------------------------------------------------------------
# Tracking Events
# ---------------------------------------------------------------------------

def log_tracking_event(
    db: Session,
    animal_id: int,
    tracking_id: int,
    pos: tuple[float, float],
    speed_mps: float = 0.0,
    distance_travelled_m: float = 0.0,
    camera_id: Optional[str] = None,
    gps: Optional[tuple[float, float]] = None,
    heading_deg: Optional[float] = None,
) -> TrackingEvent:
    trk = TrackingEvent(
        animal_id=animal_id,
        tracking_id=tracking_id,
        pos_x=pos[0],
        pos_y=pos[1],
        speed_mps=speed_mps,
        distance_travelled_m=distance_travelled_m,
        camera_id=camera_id,
        gps_lat=gps[0] if gps else None,
        gps_lon=gps[1] if gps else None,
        heading_deg=heading_deg,
    )
    db.add(trk)
    db.commit()
    db.refresh(trk)
    return trk


def get_animal_trajectory(
    db: Session, animal_id: int, limit: int = 500
) -> Sequence[TrackingEvent]:
    return (
        db.query(TrackingEvent)
        .filter(TrackingEvent.animal_id == animal_id)
        .order_by(TrackingEvent.timestamp.asc())
        .limit(limit)
        .all()
    )


# ---------------------------------------------------------------------------
# Behaviors
# ---------------------------------------------------------------------------

def log_behavior(
    db: Session,
    animal_id: int,
    behavior: str,
    confidence: float,
    duration_seconds: float = 1.0,
    camera_id: Optional[str] = None,
) -> BehaviorRecord:
    rec = BehaviorRecord(
        animal_id=animal_id,
        behavior=behavior,
        confidence=confidence,
        duration_seconds=duration_seconds,
        camera_id=camera_id,
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)
    return rec


def list_behaviors(
    db: Session, animal_id: Optional[int] = None, limit: int = 100
) -> Sequence[BehaviorRecord]:
    q = db.query(BehaviorRecord)
    if animal_id:
        q = q.filter(BehaviorRecord.animal_id == animal_id)
    return q.order_by(desc(BehaviorRecord.timestamp)).limit(limit).all()


# ---------------------------------------------------------------------------
# Health Records
# ---------------------------------------------------------------------------

def log_health_record(
    db: Session,
    animal_id: int,
    activity_level: float,
    stress_index: float,
    rest_min: float = 0.0,
    sleeping_min: float = 0.0,
    feeding_min: float = 0.0,
    walking_dist_m: float = 0.0,
    daily_movement_m: float = 0.0,
    injury_risk: float = 0.0,
    diagnostics: Optional[str] = None,
) -> HealthRecord:
    rec = HealthRecord(
        animal_id=animal_id,
        activity_level=activity_level,
        stress_index=stress_index,
        rest_duration_min=rest_min,
        sleeping_duration_min=sleeping_min,
        feeding_duration_min=feeding_min,
        walking_distance_m=walking_dist_m,
        daily_movement_m=daily_movement_m,
        injury_risk_score=injury_risk,
        diagnostics=diagnostics,
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)
    return rec


def get_latest_health(db: Session, animal_id: int) -> Optional[HealthRecord]:
    return (
        db.query(HealthRecord)
        .filter(HealthRecord.animal_id == animal_id)
        .order_by(desc(HealthRecord.timestamp))
        .first()
    )


# ---------------------------------------------------------------------------
# Anomalies
# ---------------------------------------------------------------------------

def log_anomaly(
    db: Session,
    anomaly_type: str,
    anomaly_score: float,
    animal_id: Optional[int] = None,
    severity: Any = AlertSeverity.MEDIUM,
    confidence: float = 0.8,
    camera_id: Optional[str] = None,
    details: Optional[str] = None,
) -> AnomalyEvent:
    str_sev = severity.value if hasattr(severity, "value") else str(severity)
    event = AnomalyEvent(
        animal_id=animal_id,
        anomaly_type=anomaly_type,
        anomaly_score=anomaly_score,
        severity=str_sev,
        confidence=confidence,
        camera_id=camera_id,
        details=details,
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    return event


def list_anomalies(db: Session, limit: int = 100) -> Sequence[AnomalyEvent]:
    return db.query(AnomalyEvent).order_by(desc(AnomalyEvent.timestamp)).limit(limit).all()


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

def create_alert(
    db: Session,
    alert_type: Any,
    severity: Any,
    message: str,
    animal_id: Optional[int] = None,
    camera_id: Optional[str] = None,
    zone_name: Optional[str] = None,
    gps: Optional[tuple[float, float]] = None,
    snapshot_path: Optional[str] = None,
    video_path: Optional[str] = None,
) -> Alert:
    str_type = alert_type.value if hasattr(alert_type, "value") else str(alert_type)
    str_sev = severity.value if hasattr(severity, "value") else str(severity)
    alert = Alert(
        animal_id=animal_id,
        alert_type=str_type,
        severity=str_sev,
        message=message,
        camera_id=camera_id,
        zone_name=zone_name,
        gps_lat=gps[0] if gps else None,
        gps_lon=gps[1] if gps else None,
        snapshot_path=snapshot_path,
        video_path=video_path,
    )
    db.add(alert)
    db.commit()
    db.refresh(alert)
    logger.warning(f"ALERT CREATED [{str_sev}] {str_type}: {message}")
    return alert


def list_alerts(
    db: Session,
    severity: Optional[Any] = None,
    unacknowledged_only: bool = False,
    limit: int = 100,
) -> Sequence[Alert]:
    q = db.query(Alert)
    if severity:
        str_sev = severity.value if hasattr(severity, "value") else str(severity)
        q = q.filter(Alert.severity == str_sev)
    if unacknowledged_only:
        q = q.filter(Alert.acknowledged.is_(False))
    return q.order_by(desc(Alert.timestamp)).limit(limit).all()


def acknowledge_alert(db: Session, alert_id: int) -> Optional[Alert]:
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if alert:
        alert.acknowledged = True
        db.commit()
        db.refresh(alert)
    return alert


# ---------------------------------------------------------------------------
# Cameras & Sources
# ---------------------------------------------------------------------------

def register_camera(
    db: Session,
    camera_id: str,
    name: str,
    url: str,
    camera_type=None,
    fps: float = 30.0,
    resolution: tuple[int, int] = (1280, 720),
    gps: Optional[tuple[float, float, float]] = None,
) -> Camera:
    from database.models import CameraType
    c_type = camera_type or CameraType.RTSP
    cam = db.query(Camera).filter(Camera.camera_id == camera_id).first()
    now = dt.datetime.now(dt.timezone.utc)
    if cam is None:
        cam = Camera(
            camera_id=camera_id,
            name=name,
            url=url,
            camera_type=c_type,
            fps=fps,
            resolution_w=resolution[0],
            resolution_h=resolution[1],
            gps_lat=gps[0] if gps else None,
            gps_lon=gps[1] if gps else None,
            gps_alt=gps[2] if gps and len(gps) > 2 else None,
            last_ping=now,
        )
        db.add(cam)
    else:
        cam.name = name
        cam.url = url
        cam.fps = fps
        cam.last_ping = now
    db.commit()
    db.refresh(cam)
    return cam


def list_cameras(db: Session) -> Sequence[Camera]:
    return db.query(Camera).all()


# ---------------------------------------------------------------------------
# Dashboard Analytics Summaries
# ---------------------------------------------------------------------------

def get_dashboard_counts(db: Session) -> Dict[str, Any]:
    total_animals = db.query(func.count(Animal.id)).scalar() or 0
    active_animals = db.query(func.count(Animal.id)).filter(Animal.active.is_(True)).scalar() or 0
    total_detections = db.query(func.count(Detection.id)).scalar() or 0
    active_alerts = db.query(func.count(Alert.id)).filter(Alert.acknowledged.is_(False)).scalar() or 0
    critical_alerts = (
        db.query(func.count(Alert.id))
        .filter(Alert.severity == AlertSeverity.CRITICAL, Alert.acknowledged.is_(False))
        .scalar()
        or 0
    )
    total_cameras = db.query(func.count(Camera.id)).scalar() or 0

    return {
        "total_animals": total_animals,
        "active_animals": active_animals,
        "total_detections": total_detections,
        "active_alerts": active_alerts,
        "critical_alerts": critical_alerts,
        "total_cameras": total_cameras,
    }


def get_species_counts(db: Session) -> Dict[str, int]:
    rows = db.query(Animal.species, func.count(Animal.id)).group_by(Animal.species).all()
    return {species: count for species, count in rows}
