"""
Health API Router.
Provides GET /health (system health diagnostics) and GET /health/records (wildlife health statistics).
"""

from typing import List, Dict, Any, Optional
import time
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import text

from database.database import get_db
from database.models import HealthRecord
from database.crud import list_animals, get_latest_health
from utils.device import get_device_info
from health.health_monitor import HealthMonitor

router = APIRouter(prefix="", tags=["Health"])

_health_monitor = HealthMonitor()


@router.get("/health")
def get_system_health(db: Session = Depends(get_db)):
    """
    Returns system health status: database connectivity, hardware acceleration, and system uptime.
    """
    db_ok = False
    try:
        db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False

    device_info = get_device_info()

    return {
        "status": "healthy" if db_ok else "degraded",
        "database_connected": db_ok,
        "device": device_info,
        "service": "AI Wildlife Animal Monitoring Backend",
        "version": "1.0.0",
        "timestamp": time.time(),
    }


@router.get("/health/records")
def get_health_records(
    animal_id: Optional[int] = Query(None),
    limit: int = Query(50),
    db: Session = Depends(get_db),
):
    """
    Returns historical observable health records for tracked animals.
    """
    q = db.query(HealthRecord)
    if animal_id is not None:
        q = q.filter(HealthRecord.animal_id == animal_id)

    records = q.order_by(HealthRecord.timestamp.desc()).limit(limit).all()

    return [
        {
            "id": r.id,
            "animal_id": r.animal_id,
            "timestamp": r.timestamp.isoformat(),
            "activity_level": round(r.activity_level, 3),
            "stress_index": round(r.stress_index, 3),
            "rest_duration_min": r.rest_duration_min,
            "sleeping_duration_min": r.sleeping_duration_min,
            "feeding_duration_min": r.feeding_duration_min,
            "walking_distance_m": r.walking_distance_m,
            "daily_movement_m": r.daily_movement_m,
            "injury_risk_score": r.injury_risk_score,
            "diagnostics": r.diagnostics,
        }
        for r in records
    ]
