"""
AI Wildlife Animal Monitoring System - Central Application Entrypoint.

Supports CLI modes:
    python main.py                       # Runs default monitoring pipeline
    python main.py --mode api            # Starts FastAPI backend server
    python main.py --mode dashboard      # Starts Streamlit executive dashboard
    python main.py --mode camera         # Runs live camera feed (USB / RTSP)
    python main.py --mode video          # Runs video analysis pipeline
    python main.py --init-db             # Initializes database tables & schema
"""

from __future__ import annotations
import argparse
import sys
import os
import subprocess
import time
from typing import Optional

import cv2
import numpy as np

from utils.config_loader import load_config, resolve_path
from utils.logger import get_logger
from utils.device import select_device, get_device_info
from detection.detector import get_detector
from tracking.tracker import get_tracker
from behavior.behavior_classifier import BehaviorClassifier
from anomaly.anomaly_detector import AnomalyDetector
from cameras.camera_manager import CameraManager
from cameras.video_source import BaseVideoSource
from utils.video_sources import get_video_source
from database.database import init_db, session_scope
from database.crud import get_or_create_animal, log_detection, log_tracking_event, log_behavior, log_anomaly, create_alert
from database.models import AlertType, AlertSeverity
from utils.image import draw_bounding_box, draw_trajectory, draw_fence_polygon, get_color_for_id

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Autonomous AI Wildlife Animal Monitoring & Surveillance System"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="pipeline",
        choices=["pipeline", "api", "dashboard", "camera", "video"],
        help="Execution mode: pipeline (default), api, dashboard, camera, or video",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        choices=["usb", "rtsp", "ip_camera", "drone", "video_file", "image_folder"],
        help="Video source type. Defaults to configs/config.yaml -> video_sources.default",
    )
    parser.add_argument("--video-path", type=str, default=None, help="Path to video file if mode=video")
    parser.add_argument("--camera-id", type=str, default="cam_main", help="Logical camera identifier")
    parser.add_argument("--tracker", type=str, default="bytetrack", choices=["bytetrack", "deepsort"], help="Tracker engine")
    parser.add_argument("--model", type=str, default=None, help="YOLO checkpoint or type (yolo11 / yolov8)")
    parser.add_argument("--confidence", type=float, default=None, help="Confidence threshold override")
    parser.add_argument("--device", type=str, default="auto", help="auto, cuda, cuda:0, or cpu")
    parser.add_argument("--no-display", action="store_true", help="Disable preview GUI window")
    parser.add_argument("--no-db", action="store_true", help="Disable writing results to PostgreSQL")
    parser.add_argument("--save-video", type=str, default=None, help="Save annotated output video path")
    parser.add_argument("--init-db", action="store_true", help="Initialize database schema and exit")
    parser.add_argument("--port", type=int, default=None, help="Port override for API or Dashboard")
    return parser.parse_args()


def run_api_server(port: Optional[int] = None):
    """Starts the FastAPI backend server using uvicorn."""
    import uvicorn
    from api.main import app
    cfg = load_config()
    server_port = port or cfg.api.port
    server_host = cfg.api.host
    logger.info(f"Launching FastAPI backend on http://{server_host}:{server_port} (Swagger docs: http://{server_host}:{server_port}/docs)")
    uvicorn.run(app, host=server_host, port=server_port)


def run_dashboard(port: Optional[int] = None):
    """Launches the Streamlit executive dashboard on Port 3000."""
    dashboard_script = os.path.abspath(os.path.join(os.path.dirname(__file__), "dashboard", "app.py"))
    port_to_use = port or 3000
    cmd = [sys.executable, "-m", "streamlit", "run", dashboard_script, "--server.port", str(port_to_use), "--server.headless", "true"]
    logger.info(f"Launching Wildlife Frontend on Port {port_to_use}: {' '.join(cmd)}")
    subprocess.run(cmd)


def run_surveillance_pipeline(args: argparse.Namespace):
    """
    Executes the end-to-end multi-object tracking, behavior recognition,
    and anomaly detection pipeline.
    """
    cfg = load_config()
    source_type = args.source or ("usb" if args.mode == "camera" else cfg.video_sources.default)
    if args.mode == "video":
        source_type = "video_file"

    logger.info(f"Initializing Wildlife Monitoring Engine (source={source_type}, tracker={args.tracker})...")

    # Hardware diagnostics
    dev = select_device(args.device)
    logger.info(f"Compute acceleration device: {dev}")

    # Initialize AI Detectors, Trackers, Behavior & Anomaly modules
    conf = args.confidence or cfg.detection.confidence_threshold
    detector = get_detector(
        model_type=args.model,
        confidence_threshold=conf,
        device=args.device,
    )
    tracker = get_tracker(args.tracker)
    behavior_classifier = BehaviorClassifier(device=str(dev))
    anomaly_detector = AnomalyDetector()

    # Video Source
    if args.video_path:
        from cameras.video_source import VideoFile
        source = VideoFile(source_id=args.camera_id, file_path=args.video_path, loop=False)
    else:
        source = get_video_source(cfg, source_type=source_type)

    # Virtual fence enclosure polygon
    FENCE_PERIMETER = [[50, 50], [1230, 50], [1230, 670], [50, 670]]

    # Video Writer
    writer = None
    frame_count = 0
    total_detections = 0
    t_start = time.time()

    logger.info(f"Surveillance pipeline active. Processing frames...")
    if not args.no_display:
        logger.info("Preview display enabled. Press 'q' in the window to stop.")

    try:
        with source:
            for frame_id, frame, timestamp in source:
                frame_count += 1

                # 1. Detection
                detections = detector.detect(
                    frame=frame,
                    confidence_threshold=conf,
                    frame_number=frame_count,
                    timestamp=timestamp,
                )
                total_detections += len(detections)

                # 2. Tracking
                tracks = tracker.update(
                    detections=detections,
                    frame=frame,
                    frame_number=frame_count,
                    timestamp=timestamp,
                )

                # 3. Annotations & Analytics
                annotated = frame.copy()
                annotated = draw_fence_polygon(annotated, FENCE_PERIMETER, zone_name="Sanctuary Perimeter", alpha=0.15)

                for trk in tracks:
                    # Enclosure check
                    is_inside = (
                        FENCE_PERIMETER[0][0] <= trk.current_position[0] <= FENCE_PERIMETER[1][0]
                        and FENCE_PERIMETER[0][1] <= trk.current_position[1] <= FENCE_PERIMETER[2][1]
                    )
                    fence_dist = 50.0 if is_inside else -10.0

                    # Behavior classification
                    beh_result = behavior_classifier.update_and_classify(
                        tracking_id=trk.tracking_id,
                        curr_pos=trk.current_position,
                        prev_pos=trk.previous_position,
                        prev_prev_pos=None,
                        bbox=trk.bbox,
                        speed_mps=trk.speed,
                    )
                    beh_label = beh_result["behavior"]

                    # Anomaly evaluation
                    anom_result = anomaly_detector.evaluate(
                        tracking_id=trk.tracking_id,
                        speed_history=[trk.speed],
                        aspect_ratios=[1.2],
                        distance_to_fence_m=fence_dist,
                        behavior_type=beh_label,
                    )

                    # Drawing
                    box_color = (0, 0, 255) if not is_inside else get_color_for_id(trk.tracking_id)
                    label = f"#{trk.tracking_id} {trk.species} [{beh_label}] {trk.speed:.1f}m/s"
                    annotated = draw_bounding_box(annotated, trk.bbox, label, color=box_color)

                    if trk.trajectory:
                        annotated = draw_trajectory(annotated, trk.trajectory, color=box_color)

                    # Database Logging
                    if not args.no_db:
                        try:
                            with session_scope() as db:
                                uid = f"uid_{trk.species}_{trk.tracking_id}"
                                animal = get_or_create_animal(db, track_uid=uid, species=trk.species)
                                log_tracking_event(
                                    db=db,
                                    animal_id=animal.id,
                                    tracking_id=trk.tracking_id,
                                    pos=trk.current_position,
                                    speed_mps=trk.speed,
                                    distance_travelled_m=trk.distance_travelled,
                                    camera_id=args.camera_id,
                                )
                                log_behavior(
                                    db=db,
                                    animal_id=animal.id,
                                    behavior=beh_label,
                                    confidence=beh_result["confidence"],
                                    camera_id=args.camera_id,
                                )
                                if anom_result and anom_result["severity"] in (AlertSeverity.HIGH, AlertSeverity.CRITICAL):
                                    create_alert(
                                        db=db,
                                        animal_id=animal.id,
                                        alert_type=AlertType[anom_result["anomaly_type"]] if anom_result["anomaly_type"] in AlertType.__members__ else AlertType.GENERIC,
                                        severity=anom_result["severity"],
                                        message=anom_result["details"],
                                        camera_id=args.camera_id,
                                    )
                        except Exception as db_err:
                            logger.error(f"Database logging error: {db_err}")

                # Save video
                if args.save_video:
                    if writer is None:
                        out_path = resolve_path(args.save_video)
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        h, w = annotated.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(str(out_path), fourcc, source.fps, (w, h))
                    writer.write(annotated)

                # GUI Display
                if not args.no_display:
                    cv2.imshow("Wildlife AI Surveillance", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        logger.info("Quitting surveillance loop on user request.")
                        break

                # FPS benchmark log
                if frame_count % 30 == 0:
                    elapsed = time.time() - t_start
                    fps = frame_count / elapsed if elapsed > 0 else 0.0
                    logger.info(f"Frame {frame_count:04d} | Active Tracks: {len(tracks):02d} | Pipeline FPS: {fps:.1f}")

    finally:
        if writer:
            writer.release()
            logger.info(f"Saved annotated recording to {args.save_video}")
        if not args.no_display:
            cv2.destroyAllWindows()


def main():
    args = parse_args()

    if args.init_db:
        init_db()
        print("Database initialized successfully.")
        return

    if args.mode == "api":
        run_api_server(port=args.port)
    elif args.mode == "dashboard":
        run_dashboard(port=args.port)
    else:
        # pipeline / camera / video
        run_surveillance_pipeline(args)


if __name__ == "__main__":
    main()
