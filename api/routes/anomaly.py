"""
Anomaly API Router.
Provides POST /anomaly for scoring anomalies and generating high-severity alerts.
"""

from typing import Dict, Any, Optional
import time
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from database.database import get_db
from database.crud import log_anomaly, create_alert, get_animal_by_id
from database.models import AlertType, AlertSeverity
from anomaly.anomaly_detector import AnomalyDetector
from database.schemas import AnomalyInferenceRequest

router = APIRouter(prefix="", tags=["Anomaly"])

_anomaly_detector = AnomalyDetector()


@router.post("/anomaly")
def evaluate_anomaly(
    request: AnomalyInferenceRequest,
    db: Session = Depends(get_db),
):
    """
    Evaluates tracking features for safety anomalies, fence breaches, and distress.
    """
    animal_id = request.animal_id
    speed_val = request.features[0] if request.features else 0.0

    eval_result = _anomaly_detector.evaluate(
        tracking_id=animal_id or 1,
        speed_history=[speed_val, speed_val * 1.1],
        aspect_ratios=[1.2, 1.25],
        distance_to_fence_m=request.features[6] * 100.0 if len(request.features) > 6 else 50.0,
    )

    if eval_result:
        # Log anomaly to database
        event = log_anomaly(
            db=db,
            animal_id=animal_id,
            anomaly_type=eval_result["anomaly_type"],
            anomaly_score=eval_result["anomaly_score"],
            severity=eval_result["severity"],
            confidence=eval_result["confidence"],
            details=eval_result["details"],
        )

        # Generate corresponding operational alert if severity is HIGH or CRITICAL
        if eval_result["severity"] in (AlertSeverity.HIGH, AlertSeverity.CRITICAL):
            create_alert(
                db=db,
                animal_id=animal_id,
                alert_type=AlertType[eval_result["anomaly_type"]] if eval_result["anomaly_type"] in AlertType.__members__ else AlertType.GENERIC,
                severity=eval_result["severity"],
                message=eval_result["details"],
            )

        return {
            "status": "anomaly_detected",
            "anomaly_type": eval_result["anomaly_type"],
            "anomaly_score": eval_result["anomaly_score"],
            "severity": eval_result["severity"].value,
            "confidence": eval_result["confidence"],
            "details": eval_result["details"],
            "timestamp": eval_result["timestamp"],
        }

    return {
        "status": "nominal",
        "anomaly_score": 0.05,
        "message": "Animal movement parameters within expected baseline.",
        "timestamp": time.time(),
    }
