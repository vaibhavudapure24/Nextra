"""
Streamlit Page: Movement Heatmaps.
Visualizes 2D animal movement density and migration corridors across the reserve.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import cv2

from dashboard.theme import apply_theme
from database.database import session_scope
from database.models import Animal, TrackingEvent
from tracking.trajectory import TrajectoryManager

st.set_page_config(page_title="Movement Heatmap", page_icon="🗺️", layout="wide")
apply_theme()

st.title("🗺️ Wildlife Movement & Density Heatmaps")
st.markdown("Analyze spatial utilization, animal corridors, and congregation zones from trajectory coordinates.")

# Filter sidebar
st.sidebar.header("Heatmap Filters")
with session_scope() as db:
    animals = db.query(Animal).all()
    animal_options = {f"Animal #{a.id} ({a.species})": a.id for a in animals}

selected_animal = st.sidebar.selectbox("Select Animal Focus", ["All Animals"] + list(animal_options.keys()))
blur_radius = st.sidebar.slider("Kernel Density Radius", min_value=5, max_value=40, value=20)
colormap_choice = st.sidebar.selectbox("Heatmap Palette", ["Viridis", "Inferno", "Hot", "Jet"])

# Fetch tracking coordinates
with session_scope() as db:
    q = db.query(TrackingEvent)
    if selected_animal != "All Animals":
        target_id = animal_options[selected_animal]
        q = q.filter(TrackingEvent.animal_id == target_id)
    events = q.limit(2000).all()

if events:
    xs = [e.pos_x for e in events]
    ys = [e.pos_y for e in events]
    speeds = [e.speed_mps for e in events]

    fig = px.density_heatmap(
        x=xs,
        y=ys,
        z=speeds,
        nbinsx=50,
        nbinsy=50,
        color_continuous_scale=colormap_choice.lower(),
        title="2D Enclosure Spatial Density Field (Weighted by Velocity)",
        labels={"x": "X Coordinate (Pixels / Meters)", "y": "Y Coordinate (Pixels / Meters)", "z": "Kinetic Energy"},
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ecf0f1"),
        height=600,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary metrics
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Analyzed Trajectory Points", len(events))
    with m2:
        st.metric("Average Movement Speed", f"{np.mean(speeds):.2f} m/s")
    with m3:
        st.metric("Max Speed Recorded", f"{np.max(speeds):.2f} m/s")

else:
    # Synthetic demonstration when tracking table is freshly initialized
    st.info("No tracking points in database yet. Displaying simulated enclosure movement corridor.")
    np.random.seed(42)
    t = np.linspace(0, 10, 400)
    sim_x = 640 + 250 * np.cos(t) + np.random.normal(0, 20, 400)
    sim_y = 360 + 180 * np.sin(2 * t) + np.random.normal(0, 20, 400)

    fig = px.density_heatmap(
        x=sim_x,
        y=sim_y,
        nbinsx=40,
        nbinsy=40,
        color_continuous_scale="inferno",
        title="Simulated Animal Corridor Density (North Enclosure)",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ecf0f1"),
        height=550,
    )
    st.plotly_chart(fig, use_container_width=True)
