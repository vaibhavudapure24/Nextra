"""
Tracking API Router.
Provides POST /track for multi-object animal tracking (ByteTrack & DeepSORT).
"""

from typing import List, Dict, Any, Optional
import time
from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session

from database.database import get_db
from database.crud import get_or_create_animal, log_tracking_event
from tracking.tracker import get_tracker, TrackedObject
from detection.postprocessing import DetectionResult
from database.schemas import TrackRequest

router = APIRouter(prefix="", tags=["Tracking"])

# Shared tracker instances per type
_trackers = {
    "bytetrack": get_tracker("bytetrack"),
    "deepsort": get_tracker("deepsort"),
}


@router.post("/track")
def track_animals(
    request: TrackRequest,
    db: Session = Depends(get_db),
):
    """
    Associates detections across frames using ByteTrack or DeepSORT.
    """
    tracker_name = request.tracker_type.lower() if request.tracker_type else "bytetrack"
    tracker = _trackers.get(tracker_name)
    if not tracker:
        tracker = get_tracker(tracker_name)
        _trackers[tracker_name] = tracker

    # Convert incoming dicts to DetectionResults
    det_objects = []
    for d in request.detections:
        bbox = (
            float(d.get("x1", 0.0)),
            float(d.get("y1", 0.0)),
            float(d.get("x2", 0.0)),
            float(d.get("y2", 0.0)),
        )
        det_obj = DetectionResult.create(
            class_id=int(d.get("class_id", 0)),
            species=str(d.get("species", "animal")),
            confidence=float(d.get("confidence", 0.8)),
            bbox=bbox,
        )
        det_objects.append(det_obj)

    tracks: List[TrackedObject] = tracker.update(det_objects)

    # Persist tracking events and ensure animal identity exists
    results = []
    for trk in tracks:
        track_uid = f"uid_{trk.species}_{trk.tracking_id}"
        animal = get_or_create_animal(db, track_uid=track_uid, species=trk.species)
        log_tracking_event(
            db=db,
            animal_id=animal.id,
            tracking_id=trk.tracking_id,
            pos=trk.current_position,
            speed_mps=trk.speed,
            distance_travelled_m=trk.distance_travelled,
            camera_id=request.camera_id,
        )
        results.append(trk.to_dict())

    return {
        "tracker": tracker_name,
        "active_tracks_count": len(tracks),
        "tracks": results,
        "timestamp": time.time(),
    }
