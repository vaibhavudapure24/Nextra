"""
Behavior API Router.
Provides POST /behavior for sequence-based behavior recognition.
"""

from typing import Dict, Any, Optional
import time
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from database.database import get_db
from database.crud import log_behavior, get_animal_by_id
from behavior.behavior_classifier import BehaviorClassifier
from database.schemas import BehaviorInferenceRequest

router = APIRouter(prefix="", tags=["Behavior"])

_classifier = BehaviorClassifier()


@router.post("/behavior")
def classify_animal_behavior(
    request: BehaviorInferenceRequest,
    db: Session = Depends(get_db),
):
    """
    Classifies temporal animal behavior from recent sequence kinematics.
    """
    animal = get_animal_by_id(db, request.animal_id)
    if not animal:
        raise HTTPException(status_code=404, detail=f"Animal ID {request.animal_id} not found.")

    # Use classifier
    result = _classifier.update_and_classify(
        tracking_id=request.animal_id,
        curr_pos=(640.0, 360.0),
        prev_pos=(638.0, 360.0),
        prev_prev_pos=(636.0, 360.0),
        bbox=(600.0, 300.0, 680.0, 420.0),
        speed_mps=1.2,
    )

    rec = log_behavior(
        db=db,
        animal_id=animal.id,
        behavior=result["behavior"],
        confidence=result["confidence"],
    )

    return {
        "animal_id": animal.id,
        "species": animal.species,
        "behavior": result["behavior"],
        "confidence": result["confidence"],
        "is_ai_model": result["is_ai_model"],
        "model_type": result["model_type"],
        "timestamp": time.time(),
    }
