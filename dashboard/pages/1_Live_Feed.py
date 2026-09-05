"""
Streamlit Page: Live Surveillance Feed.
Real-time wildlife detection, multi-object tracking (ByteTrack / DeepSORT),
behavior classification, virtual fence boundary enforcement, and night-vision enhancement.
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import cv2
import numpy as np
import streamlit as st

from dashboard.theme import apply_theme
from dashboard.state import get_config
from detection.detector import get_detector
from tracking.tracker import get_tracker
from behavior.behavior_classifier import BehaviorClassifier
from anomaly.anomaly_detector import AnomalyDetector
from cameras.camera_manager import CameraManager
from utils.image import enhance_night_vision, draw_fence_polygon, draw_trajectory, draw_bounding_box, get_color_for_id
from utils.video_sources import get_video_source
from database.database import session_scope
from database.crud import get_or_create_animal, log_detection, log_tracking_event, log_behavior, create_alert
from database.models import AlertType, AlertSeverity

st.set_page_config(page_title="Live Surveillance Feed", page_icon="🎥", layout="wide")
apply_theme()

st.title("🎥 Live Autonomous Surveillance & Tracking")
cfg = get_config()

with st.sidebar:
    st.header("Surveillance Controls")
    source_label = st.selectbox("Camera Source", ["Video File", "Webcam (USB)", "Image Folder"])
    source_map = {"Video File": "video_file", "Webcam (USB)": "usb", "Image Folder": "image_folder"}

    camera_index = 0
    if source_label == "Webcam (USB)":
        camera_index = st.number_input("Webcam Device Index (Linux usually 0 or 2)", min_value=0, max_value=10, value=0)

    tracker_type = st.selectbox("Tracking Algorithm", ["ByteTrack", "DeepSORT"], index=0)
    confidence = st.slider("Detection Confidence Threshold", 0.10, 0.95, 0.40, 0.05)

    st.subheader("Visual Overlays")
    show_trails = st.checkbox("Show Motion Trajectory Trails", value=True)
    show_fence = st.checkbox("Show Virtual Enclosure Fence", value=True)
    night_vision = st.checkbox("Low-Light / Night Vision Enhancement", value=False)
    log_to_db = st.checkbox("Log Detections & Tracks to Database", value=True)

    run = st.checkbox("▶ Start Live Surveillance Stream", value=False)

# Initialize processors
detector = get_detector(confidence_threshold=confidence)
tracker = get_tracker(tracker_type.lower())
behavior_classifier = BehaviorClassifier()
anomaly_detector = AnomalyDetector()

tab_stream, tab_cam = st.tabs(["🔴 Live Surveillance Stream", "📷 Browser Live Camera Input"])

with tab_cam:
    st.write("Use your browser's webcam directly to capture and analyze animals, birds, or test objects in real time.")
    cam_picture = st.camera_input("Capture frame from local webcam")
    if cam_picture is not None:
        bytes_data = cam_picture.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        dets = detector.detect(cv2_img, confidence_threshold=confidence)
        annotated_cam = detector.draw(cv2_img, dets)
        st.image(cv2.cvtColor(annotated_cam, cv2.COLOR_BGR2RGB), caption=f"Analyzed Frame ({len(dets)} objects detected)", use_container_width=True)
        if dets:
            st.success(f"Detected: {', '.join(f'{d.species} ({d.confidence*100:.1f}%)' for d in dets)}")
        else:
            st.info("No wildlife species identified in captured frame.")

with tab_stream:
    frame_placeholder = st.empty()
    metrics_placeholder = st.empty()
    alerts_placeholder = st.empty()

# Default fence polygon
FENCE_POLYGON = [[60, 60], [1220, 60], [1220, 660], [60, 660]]

if run:
    source_type = source_map[source_label]
    try:
        if source_type == "usb":
            cfg.video_sources.usb.device_index = camera_index
            
        source = get_video_source(cfg, source_type=source_type)
        if source_type == "usb":
            source.open()
            if source.cap is None or not source.cap.isOpened():
                st.warning("⚠️ Local hardware /dev/video0 was not grabbed directly. Auto-switching to Surveillance Video Stream.")
                source = get_video_source(cfg, source_type="video_file")
    except Exception as exc:
        st.warning(f"Note: Could not open {source_type}. Falling back to Surveillance Video Feed.")
        source = get_video_source(cfg, source_type="video_file")

    frame_count = 0
    total_detections = 0
    t_start = time.time()

    with source:
        for frame_id, frame, ts in source:
            if not run:
                break

            # 1. Low light enhancement if requested
            if night_vision:
                frame = enhance_night_vision(frame, clip_limit=3.0, gamma=1.2)

            # 2. AI Detection
            detections = detector.detect(frame, confidence_threshold=confidence)
            frame_count += 1
            total_detections += len(detections)

            # 3. Tracking
            tracks = tracker.update(detections, frame=frame, frame_number=frame_count, timestamp=ts)

            # 4. Render Annotations
            annotated = frame.copy()
            
            # Draw raw detections instantly (so users see identification even before tracking locks)
            annotated = detector.draw(annotated, detections, show_tracking_id=False)

            # Virtual fence boundary
            if show_fence:
                annotated = draw_fence_polygon(
                    annotated, FENCE_POLYGON, zone_name="Sanctuary Perimeter", color=(0, 255, 0), alpha=0.15
                )

            # Render tracked animals and behaviors
            active_alerts_this_frame = []
            for trk in tracks:
                color = get_color_for_id(trk.tracking_id)
                # Check fence breach
                is_inside = (FENCE_POLYGON[0][0] <= trk.current_position[0] <= FENCE_POLYGON[1][0] and
                             FENCE_POLYGON[0][1] <= trk.current_position[1] <= FENCE_POLYGON[2][1])
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

                # Anomaly check
                anom_result = anomaly_detector.evaluate(
                    tracking_id=trk.tracking_id,
                    speed_history=[trk.speed],
                    aspect_ratios=[1.2],
                    distance_to_fence_m=fence_dist,
                    behavior_type=beh_label,
                )

                label_tag = f"#{trk.tracking_id} [{beh_label}] | {trk.speed:.1f} m/s"
                box_color = (0, 0, 255) if not is_inside else color
                annotated = draw_bounding_box(annotated, trk.bbox, label_tag, color=box_color)

                if show_trails and trk.trajectory:
                    annotated = draw_trajectory(annotated, trk.trajectory, color=box_color)

                if anom_result:
                    active_alerts_this_frame.append(f"🚨 {anom_result['anomaly_type']}: Animal #{trk.tracking_id} ({trk.species})")

                # Database logging
                if log_to_db:
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
                                camera_id="cam_main",
                            )
                            log_behavior(
                                db=db,
                                animal_id=animal.id,
                                behavior=beh_label,
                                confidence=beh_result["confidence"],
                            )
                            if anom_result and anom_result["severity"] in (AlertSeverity.HIGH, AlertSeverity.CRITICAL):
                                create_alert(
                                    db=db,
                                    animal_id=animal.id,
                                    alert_type=AlertType[anom_result["anomaly_type"]] if anom_result["anomaly_type"] in AlertType.__members__ else AlertType.GENERIC,
                                    severity=anom_result["severity"],
                                    message=anom_result["details"],
                                    camera_id="cam_main",
                                )
                    except Exception as db_err:
                        pass

            # Display frame
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(rgb, channels="RGB", use_container_width=True)

            elapsed = max(time.time() - t_start, 1e-6)
            with metrics_placeholder.container():
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Processed Frames", frame_count)
                c2.metric("Active Animals", len(tracks))
                c3.metric("Total Detections", total_detections)
                c4.metric("Real-Time FPS", f"{frame_count / elapsed:.1f}")

            with alerts_placeholder.container():
                if active_alerts_this_frame:
                    st.error(" | ".join(active_alerts_this_frame))
