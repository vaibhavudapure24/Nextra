"""
Streamlit Page: 3D Enclosure Visualization.
Renders interactive 3D terrain, virtual perimeter fences, sensor locations, and animal movement paths.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import numpy as np
import plotly.graph_objects as go

from dashboard.theme import apply_theme

st.set_page_config(page_title="3D Enclosure Map", page_icon="🌐", layout="wide")
apply_theme()

st.title("🌐 3D Sanctuary Terrain & Perimeter Boundary Map")
st.markdown("Interactive 3D spatial model displaying terrain elevation, sensor towers, virtual boundary polygons, and animal pathways.")

# Generate synthetic 3D terrain grid
grid_size = 50
x = np.linspace(-500, 500, grid_size)
y = np.linspace(-500, 500, grid_size)
X, Y = np.meshgrid(x, y)
# Natural undulating sanctuary terrain
Z = 50 * np.sin(X / 150.0) * np.cos(Y / 150.0) + 20 * np.sin(Y / 80.0)

fig = go.Figure()

# 1. Terrain Surface
fig.add_trace(go.Surface(
    z=Z, x=x, y=y,
    colorscale="Earth",
    opacity=0.85,
    showscale=False,
    name="Terrain Elevation",
))

# 2. Virtual Enclosure Fence Boundary (Polygon in 3D)
fence_x = [-400, 400, 400, -400, -400]
fence_y = [-400, -400, 400, 400, -400]
fence_z = [45, 45, 45, 45, 45]

fig.add_trace(go.Scatter3d(
    x=fence_x, y=fence_y, z=fence_z,
    mode="lines+markers",
    line=dict(color="#e74c3c", width=6),
    marker=dict(size=4, color="#e74c3c"),
    name="Perimeter Fence Line",
))

# 3. Surveillance Cameras / Tower Locations
cam_x = [-350, 350, 0]
cam_y = [-350, 350, 380]
cam_z = [65, 65, 75]
fig.add_trace(go.Scatter3d(
    x=cam_x, y=cam_y, z=cam_z,
    mode="markers+text",
    marker=dict(size=8, color="#3498db", symbol="diamond"),
    text=["Cam #1 (North)", "Cam #2 (Waterhole)", "Thermal Boundary Cam"],
    textposition="top center",
    name="Surveillance Sensors",
))

# 4. Animal Movement Paths in 3D
t = np.linspace(0, 8, 100)
path_x = 200 * np.sin(t)
path_y = 150 * np.cos(t)
path_z = 50 * np.sin(path_x / 150.0) * np.cos(path_y / 150.0) + 10

fig.add_trace(go.Scatter3d(
    x=path_x, y=path_y, z=path_z,
    mode="lines+markers",
    line=dict(color="#2ecc71", width=4),
    marker=dict(size=3, color="#2ecc71"),
    name="Animal #104 Trajectory (Elephant)",
))

# Current Animal Position
fig.add_trace(go.Scatter3d(
    x=[path_x[-1]], y=[path_y[-1]], z=[path_z[-1]],
    mode="markers+text",
    marker=dict(size=9, color="#f1c40f", symbol="circle"),
    text=["Current: Elephant #104"],
    textposition="top center",
    name="Current Position",
))

fig.update_layout(
    scene=dict(
        xaxis_title="West-East (Meters)",
        yaxis_title="South-North (Meters)",
        zaxis_title="Altitude (Meters)",
        aspectratio=dict(x=1.5, y=1.5, z=0.5),
        camera=dict(
            eye=dict(x=1.2, y=-1.5, z=0.8)
        ),
    ),
    margin=dict(l=10, r=10, t=30, b=10),
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#ecf0f1"),
    height=650,
)

st.plotly_chart(fig, use_container_width=True)

st.info("💡 **Tip:** Click and drag to rotate the 3D model. Scroll to zoom in/out on terrain contours and fence lines.")
