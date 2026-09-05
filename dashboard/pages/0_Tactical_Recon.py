"""
Streamlit Page: Tactical C2 Wildlife Reconnaissance & Surveillance Center.
Interactive HUD, Drone Controls, 360-degree Bio-Radar, and WebRTC Live Camera.
"""

import os
import streamlit as st
import streamlit.components.v1 as components

from dashboard.theme import apply_theme

st.set_page_config(
    page_title="Tactical C2 Wildlife Recon",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)
apply_theme()

st.title("🛰️ Autonomous C2 Tactical Wildlife Reconnaissance")
col1, col2 = st.columns([3, 1])
with col1:
    st.caption("Military-grade wildlife surveillance: WebRTC camera, Drone flight HUD, 360° radar sweep, and perimeter alarms.")
with col2:
    st.markdown(
        '<a href="/app/static/index.html" target="_blank" style="display:inline-block; margin-top:8px; padding:8px 16px; background:#06b6d4; color:#0f172a; font-weight:700; border-radius:8px; text-decoration:none; text-align:center; box-shadow:0 0 15px rgba(6,182,212,0.4);">🚀 Open Fullscreen C2 HUD</a>',
        unsafe_allow_html=True
    )

st.markdown(
    '<iframe src="/app/static/index.html" allow="camera; microphone; autoplay; fullscreen; display-capture" width="100%" height="950px" style="border:1px solid #1e293b; border-radius:12px; background:#020617;"></iframe>',
    unsafe_allow_html=True
)
