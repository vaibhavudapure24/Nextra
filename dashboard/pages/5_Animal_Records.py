"""
Streamlit Page: Animal Records, Profiles, and Historical Timelines.
Displays persistent identity dossiers, RFID tags, behavior breakdowns, health vitality, and GPS paths.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.theme import apply_theme
from database.database import session_scope
from database.crud import list_animals, get_animal_by_id, get_detection_history, get_animal_trajectory, list_behaviors, get_latest_health
from database.models import Alert

st.set_page_config(page_title="Animal Records & Profiles", page_icon="🐾", layout="wide")
apply_theme()

st.title("🐾 Animal Profiles & Longitudinal Timelines")
st.markdown("Detailed dossiers for individual animals: persistent track UIDs, RFID identification, movement kinematics, behavior timeline, and health status.")

try:
    with session_scope() as db:
        animals = list_animals(db, limit=100)
except Exception as e:
    animals = []
    st.error(f"Database query notice: {e}")

if not animals:
    st.info("No animal records found in the database yet.")
    if st.button("🌱 Populate Sanctuary Demo Animals", type="primary"):
        try:
            with session_scope() as db:
                from database.crud import log_tracking_event, log_behavior, log_health_record
                import datetime as dt

                demo_species = [
                    ("TRK-104-ELEPHANT", "Elephant", "RFID-982-A41"),
                    ("TRK-108-ZEBRA", "Zebra", "RFID-441-Z12"),
                    ("TRK-112-LION", "Lion", "RFID-110-L03"),
                    ("TRK-115-GIRAFFE", "Giraffe", "RFID-772-G99"),
                ]
                for uid, spec, rfid in demo_species:
                    a = get_or_create_animal(db, track_uid=uid, species=spec)
                    a.rfid_tag = rfid
                    db.commit()
                    # Add sample trajectory
                    for i in range(5):
                        log_tracking_event(
                            db, animal_id=a.id, tracking_id=int(uid.split("-")[1]),
                            pos=(200.0 + i*30, 300.0 + i*15), speed_mps=1.5 + i*0.2,
                            distance_travelled_m=50.0 * (i+1), camera_id="cam_main"
                        )
                    # Add sample behavior
                    log_behavior(db, animal_id=a.id, behavior="grazing", confidence=0.92)
                    log_health_record(db, animal_id=a.id, activity_level=0.72, stress_index=0.14)
            st.success("Successfully populated sanctuary demo animals! Rerun page to view.")
            st.rerun()
        except Exception as err:
            st.error(f"Could not populate demo animals: {err}")
else:
    # Top Animal Selection
    animal_labels = {f"#{a.id} — {a.species} ({a.track_uid})": a.id for a in animals}
    selected_label = st.selectbox("Select Animal Profile Dossier", list(animal_labels.keys()))
    selected_id = animal_labels[selected_label]

    try:
        with session_scope() as db:
            animal = get_animal_by_id(db, selected_id)
            detections = get_detection_history(db, animal_id=selected_id, limit=300)
            trajectory = get_animal_trajectory(db, animal_id=selected_id, limit=300)
            behaviors = list_behaviors(db, animal_id=selected_id, limit=50)
            health = get_latest_health(db, selected_id)
            alerts = db.query(Alert).filter(Alert.animal_id == selected_id).order_by(Alert.timestamp.desc()).all()
    except Exception as query_err:
        st.error(f"Error loading animal dossier details: {query_err}")
        st.stop()

    if animal is None:
        st.warning("Selected animal record could not be found.")
        st.stop()

    # Animal Profile Overview Cards
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.metric("Species", animal.species)
    with p2:
        st.metric("Track UID", animal.track_uid)
    with p3:
        st.metric("RFID Tag", animal.rfid_tag or "Unassigned")
    with p4:
        first_seen_str = animal.first_seen.strftime("%Y-%m-%d %H:%M") if (animal.first_seen and hasattr(animal.first_seen, "strftime")) else "Recorded"
        st.metric("First Seen", first_seen_str)

    st.divider()

    # Tabs for Timeline, Movement, Behaviors, and Health
    tab_timeline, tab_movement, tab_behavior, tab_health = st.tabs([
        "⏳ Interactive Timeline",
        "🗺️ Movement & Trajectory",
        "🐾 Behaviors & Diet",
        "🩺 Health & Vitality",
    ])

    with tab_timeline:
        st.subheader("Event & Observation Timeline")
        timeline_events = []
        for d in detections:
            timeline_events.append({"Time": d.timestamp, "Event": "Detection", "Details": f"Confidence: {d.confidence*100:.1f}%"})
        for b in behaviors:
            timeline_events.append({"Time": b.timestamp, "Event": "Behavior", "Details": f"{b.behavior.capitalize()} (conf: {b.confidence*100:.0f}%)"})
        for a in alerts:
            atype = a.alert_type.value if hasattr(a.alert_type, "value") else str(a.alert_type)
            timeline_events.append({"Time": a.timestamp, "Event": f"ALERT: {atype}", "Details": a.message})

        if timeline_events:
            df_timeline = pd.DataFrame(timeline_events).sort_values("Time", ascending=False)
            st.dataframe(df_timeline, use_container_width=True, hide_index=True)
        else:
            st.info("No recorded timeline events for this animal yet.")

    with tab_movement:
        st.subheader("Centroid Movement Coordinates & Trajectory")
        if trajectory:
            t_df = pd.DataFrame([
                {"Time": t.timestamp, "X": t.pos_x, "Y": t.pos_y, "Speed (m/s)": t.speed_mps, "Distance (m)": t.distance_travelled_m}
                for t in trajectory
            ])
            st.line_chart(t_df, x="Time", y="Speed (m/s)")

            fig_path = px.scatter(
                t_df, x="X", y="Y", color="Speed (m/s)",
                title="Spatial Path Displacements",
                color_continuous_scale="Viridis",
            )
            fig_path.update_layout(
                yaxis=dict(autorange="reversed"),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ecf0f1"),
            )
            st.plotly_chart(fig_path, use_container_width=True)
        else:
            st.info("No trajectory points logged yet for this animal.")

    with tab_behavior:
        st.subheader("Behavior Breakdown")
        if behaviors:
            beh_counts = pd.Series([b.behavior for b in behaviors]).value_counts()
            fig_pie = px.pie(
                values=beh_counts.values,
                names=beh_counts.index,
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#ecf0f1"))
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No behavior sequence events classified for this animal.")

    with tab_health:
        st.subheader("Health & Gait Diagnosis")
        if health:
            h1, h2, h3 = st.columns(3)
            with h1:
                st.metric("Activity Level", f"{health.activity_level*100:.1f}%")
            with h2:
                st.metric("Stress Index", f"{health.stress_index*100:.1f}%")
            with h3:
                st.metric("Injury Risk Score", f"{health.injury_risk_score*100:.1f}%")

            st.write(f"**Diagnostic Notes:** {health.diagnostics or 'Nominal health profile.'}")
            st.write(f"**Rest Duration:** {health.rest_duration_min:.1f} min | **Walking Distance:** {health.walking_distance_m:.1f} m")
        else:
            st.info("Health evaluation pending. Sufficient movement sequence required.")
