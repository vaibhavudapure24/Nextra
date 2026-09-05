"""Upload a single image or short video and run detection directly - no
camera or database required. Useful for isolating model/threshold issues."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from dashboard.theme import apply_theme
from dashboard.state import get_config, get_detector

st.set_page_config(page_title="Upload and Test - AI Wildlife Monitoring", page_icon="📤", layout="wide")
apply_theme()

st.title("📤 Upload & Test Detection")
st.caption("Upload a photo or short video to test the detector directly.")

cfg = get_config()
detector = get_detector()

st.info(f"Currently loaded model: **{cfg.detection.model_path}** ({cfg.detection.model_type})")

confidence = st.slider("Confidence Threshold", 0.05, 0.95, float(cfg.detection.confidence_threshold), 0.05)

tab_img, tab_vid = st.tabs(["🖼️ Image", "🎞️ Video"])

with tab_img:
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"], key="img_upload")
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        import requests
        import io
        import os
        from detection.postprocessing import DetectionResult
        
        # Send image to Backend API for detection
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        
        api_url = os.environ.get("API_URL", f"http://{cfg.api.host}:{cfg.api.port}")
        api_key = getattr(cfg.api, "api_key", "wl_super_secret_key_123")
        
        try:
            response = requests.post(
                f"{api_url}/detect",
                headers={"X-API-Key": api_key},
                files={"file": ("image.jpg", buffered.getvalue(), "image/jpeg")},
                data={"confidence_threshold": confidence}
            )
            response.raise_for_status()
            api_data = response.json()
            api_detections = api_data.get("detections", [])
            
            # Reconstruct DetectionResults for standard rendering
            detections = []
            for d in api_detections:
                det = DetectionResult(
                    detection_id=d["detection_id"], class_id=d["class_id"], species=d["species"],
                    confidence=d["confidence"], x1=d["x1"], y1=d["y1"], x2=d["x2"], y2=d["y2"],
                    center_x=d["center_x"], center_y=d["center_y"]
                )
                detections.append(det)
                
            if hasattr(detector, "draw"):
                annotated = detector.draw(frame_bgr, detections)
            else:
                annotated = frame_bgr.copy()
        except Exception as e:
            st.error(f"Failed to connect to Backend API: {e}")
            detections = []
            annotated = frame_bgr.copy()
            
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)
        col1.image(image, caption="Original", use_container_width=True)
        col2.image(annotated_rgb, caption=f"Detected: {len(detections)} animal(s) via API", use_container_width=True)

        if detections:
            st.subheader("Detections")
            st.dataframe([d.to_dict() for d in detections], use_container_width=True)

            if st.button("💾 Save Detections to Database", key="img_save_btn"):
                try:
                    from database.database import session_scope
                    from database.crud import get_or_create_animal, log_detection

                    with session_scope() as db:
                        for det in detections:
                            pseudo_uid = f"{det.species}_{int(det.center_x // 50)}_{int(det.center_y // 50)}"
                            animal = get_or_create_animal(db, track_uid=pseudo_uid, species=det.species)
                            log_detection(
                                db, animal, det.species, det.confidence, det.bbox,
                                pos=(det.center_x, det.center_y), camera_id="upload_test_image",
                            )
                    st.success(f"Saved {len(detections)} detection(s) to the database.")
                except Exception as exc:
                    st.warning(f"Could not save to database: {exc}")
        else:
            st.warning(
                "No animals detected above this confidence threshold. Try lowering "
                "the slider, or confirm this image actually contains a class your "
                "current model was trained on."
            )

with tab_vid:
    uploaded_vid = st.file_uploader(
        "Upload a short video (processed frame-by-frame)", type=["mp4", "avi", "mov"], key="vid_upload",
    )
    max_frames = st.slider("Max frames to process (keeps this fast)", 10, 300, 90, 10)
    save_video_to_db = st.checkbox("Save all detections from this video to the database", value=False, key="vid_save_db")

    if uploaded_vid and st.button("Run Detection on Video"):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(uploaded_vid.read())
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        progress = st.progress(0)
        frame_placeholder = st.empty()
        detection_count_total = 0
        species_seen: set[str] = set()
        frame_idx = 0
        db_error_shown = False

        if save_video_to_db:
            from database.database import session_scope
            from database.crud import get_or_create_animal, log_detection

        while frame_idx < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            detections = detector.detect(frame, confidence_threshold=confidence)
            if hasattr(detector, "draw"):
                annotated = detector.draw(frame, detections)
            else:
                annotated = frame.copy()
                
            detection_count_total += len(detections)
            species_seen.update(d.species for d in detections)
            frame_placeholder.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True,
            )

            if save_video_to_db and detections:
                try:
                    with session_scope() as db:
                        for det in detections:
                            pseudo_uid = f"{det.species}_{int(det.center_x // 50)}_{int(det.center_y // 50)}"
                            animal = get_or_create_animal(db, track_uid=pseudo_uid, species=det.species)
                            log_detection(
                                db, animal, det.species, det.confidence, det.bbox,
                                pos=(det.center_x, det.center_y), camera_id="upload_test_video", frame_number=frame_idx,
                            )
                except Exception as exc:
                    if not db_error_shown:
                        st.warning(f"Database logging failed (will keep trying): {exc}")
                        db_error_shown = True

            frame_idx += 1
            progress.progress(frame_idx / max_frames)

        cap.release()
        Path(tmp_path).unlink(missing_ok=True)

        st.success(
            f"Processed {frame_idx} frames. Total detections: {detection_count_total}. "
            f"Species seen: {', '.join(sorted(species_seen)) if species_seen else 'none'}"
        )
