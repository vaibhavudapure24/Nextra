"""
Detection API Router.
Provides POST /detect for running YOLO11/YOLOv8 inference on uploaded frames or camera feeds.
"""

from typing import List, Optional
import base64
import time
import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session

from database.database import get_db
from database.crud import log_detection
from detection.detector import get_detector
from detection.postprocessing import DetectionResult

router = APIRouter(prefix="", tags=["Detection"])


@router.post("/detect")
async def detect_animals(
    file: Optional[UploadFile] = File(None),
    confidence_threshold: float = Form(0.40),
    camera_id: str = Form("cam_api"),
    model_type: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    """
    Runs YOLO animal detection on an uploaded image file.
    """
    if file is None:
        raise HTTPException(status_code=400, detail="An image file must be uploaded.")

    content = await file.read()
    nparr = np.frombuffer(content, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None or frame.size == 0:
        raise HTTPException(status_code=400, detail="Could not decode uploaded image.")

    detector = get_detector(model_type=model_type, confidence_threshold=confidence_threshold)
    detections: List[DetectionResult] = detector.detect(
        frame=frame,
        confidence_threshold=confidence_threshold,
    )

    # Persist detections to database
    records = []
    for d in detections:
        det_record = log_detection(
            db=db,
            species=d.species,
            confidence=d.confidence,
            bbox=d.bbox,
            camera_id=camera_id,
        )
        records.append(d.to_dict())

    return {
        "camera_id": camera_id,
        "count": len(detections),
        "detections": records,
        "timestamp": time.time(),
    }
