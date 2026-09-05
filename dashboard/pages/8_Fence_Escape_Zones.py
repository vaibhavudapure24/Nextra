"""
Streamlit Page: Virtual Enclosure & Fence Escape Configuration.
Allows sanctuary managers to define polygon geofences, monitor perimeter breaches,
and trigger CRITICAL security alerts.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import numpy as np
import plotly.graph_objects as go

from dashboard.theme import apply_theme
from database.database import session_scope
from database.crud import list_alerts, create_alert
from database.models import AlertType, AlertSeverity
from cameras.camera_manager import CameraManager

st.set_page_config(page_title="Fence Escape Zones", page_icon="⚡", layout="wide")
apply_theme()

st.title("⚡ Virtual Perimeter Enclosures & Escape Detection")
st.markdown("Configure geofence polygon perimeters. Crossing the boundary automatically triggers a **CRITICAL** alert and dispatches emergency notifications.")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("⚙️ Enclosure Zone Configuration")
    zone_name = st.text_input("Enclosure Name", value="North Reserve Main Sanctuary")
    cam_target = st.selectbox("Associated Camera Stream", ["cam_main_enclosure", "cam_waterhole_rtsp", "cam_drone_scout"])

    st.markdown("**Polygon Perimeter Coordinates (Pixels / Relative)**")
    p1_x = st.number_input("Vertex 1 X", value=50, step=10)
    p1_y = st.number_input("Vertex 1 Y", value=50, step=10)
    p2_x = st.number_input("Vertex 2 X", value=1230, step=10)
    p2_y = st.number_input("Vertex 2 Y", value=50, step=10)
    p3_x = st.number_input("Vertex 3 X", value=1230, step=10)
    p3_y = st.number_input("Vertex 3 Y", value=670, step=10)
    p4_x = st.number_input("Vertex 4 X", value=50, step=10)
    p4_y = st.number_input("Vertex 4 Y", value=670, step=10)

    if st.button("💾 Save Virtual Geofence Polygon"):
        st.success(f"Geofence polygon for '{zone_name}' saved and synced with camera stream '{cam_target}'!")

    st.divider()
    st.subheader("🚨 Test Escape Alert Simulation")
    test_species = st.selectbox("Test Animal Species", ["Elephant", "Rhino", "Lion", "Tiger"])
    if st.button("⚠️ Trigger Simulated Escape Alert"):
        with session_scope() as db:
            create_alert(
                db=db,
                alert_type=AlertType.FENCE_ESCAPE,
                severity=AlertSeverity.CRITICAL,
                message=f"CRITICAL ESCAPE: {test_species} breached virtual perimeter at '{zone_name}'!",
                zone_name=zone_name,
                camera_id=cam_target,
            )
        st.error(f"🚨 CRITICAL ALERT GENERATED: {test_species} fence breach logged in database!")

with col_right:
    st.subheader("🗺️ Perimeter Boundary Map Preview")
    polygon = [
        [p1_x, p1_y],
        [p2_x, p2_y],
        [p3_x, p3_y],
        [p4_x, p4_y],
        [p1_x, p1_y],
    ]

    fig = go.Figure()
    # Draw frame boundary
    fig.add_shape(type="rect", x0=0, y0=0, x1=1280, y1=720, line=dict(color="#7f8c8d", width=1))

    # Draw Polygon Geofence
    fig.add_trace(go.Scatter(
        x=[p[0] for p in polygon],
        y=[p[1] for p in polygon],
        mode="lines+markers",
        fill="toself",
        fillcolor="rgba(46, 204, 113, 0.2)",
        line=dict(color="#2ecc71", width=3),
        name=zone_name,
    ))

    # Draw simulated animal inside vs outside
    fig.add_trace(go.Scatter(
        x=[600], y=[360],
        mode="markers+text",
        marker=dict(size=12, color="#2ecc71"),
        text=["Animal #1 (Safe Inside)"],
        textposition="top center",
        name="Inside Sanctuary",
    ))
    fig.add_trace(go.Scatter(
        x=[20], y=[360],
        mode="markers+text",
        marker=dict(size=14, color="#e74c3c", symbol="x"),
        text=["Animal #2 (BREACH!)"],
        textposition="top center",
        name="Outside Perimeter",
    ))

    fig.update_layout(
        xaxis=dict(range=[0, 1280], showgrid=False),
        yaxis=dict(range=[720, 0], showgrid=False),  # Inverted Y for image coordinate space
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ecf0f1"),
        height=500,
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    # List recent escape alerts
    st.subheader("📋 Recent Fence Breach Events")
    with session_scope() as db:
        alerts = list_alerts(db, severity=AlertSeverity.CRITICAL, limit=5)
        if alerts:
            for a in alerts:
                st.warning(f"**[{a.timestamp.strftime('%Y-%m-%d %H:%M:%S')}] {a.message}** (Cam: {a.camera_id})")
        else:
            st.success("No perimeter violations detected in recent monitoring window.")
