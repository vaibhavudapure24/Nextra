"""
History API Router.
Provides GET /history for querying historical detections, tracking events, and alerts.
"""

from typing import List, Dict, Any, Optional
import datetime as dt
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from database.database import get_db
from database.crud import get_detection_history, list_animals, list_alerts

router = APIRouter(prefix="", tags=["History"])


@router.get("/history")
def get_historical_records(
    species: Optional[str] = Query(None),
    animal_id: Optional[int] = Query(None),
    camera_id: Optional[str] = Query(None),
    limit: int = Query(100),
    db: Session = Depends(get_db),
):
    """
    Retrieves historical detections and movement events filtered by species or animal.
    """
    detections = get_detection_history(
        db=db,
        animal_id=animal_id,
        species=species,
        camera_id=camera_id,
        limit=limit,
    )

    return {
        "count": len(detections),
        "detections": [
            {
                "id": d.id,
                "animal_id": d.animal_id,
                "species": d.species,
                "confidence": round(d.confidence, 4),
                "camera_id": d.camera_id,
                "bbox": [d.bbox_x1, d.bbox_y1, d.bbox_x2, d.bbox_y2],
                "center": [d.center_x, d.center_y],
                "gps": [d.gps_lat, d.gps_lon] if d.gps_lat else None,
                "timestamp": d.timestamp.isoformat(),
            }
            for d in detections
        ],
    }
