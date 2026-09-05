"""Shared dark-theme styling for every dashboard page."""

import streamlit as st

_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&family=JetBrains+Mono:wght@400;600&family=Plus+Jakarta+Sans:wght@400;600;700&display=swap');

    .stApp {
        background-color: #06090e;
        background-image: 
            radial-gradient(circle at 50% 0%, rgba(6, 182, 212, 0.08) 0%, transparent 50%),
            linear-gradient(rgba(15, 23, 42, 0.4) 1px, transparent 1px),
            linear-gradient(90deg, rgba(15, 23, 42, 0.4) 1px, transparent 1px);
        background-size: 100% 100%, 32px 32px, 32px 32px;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif !important;
        letter-spacing: 0.05em;
        color: #f8fafc !important;
    }

    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.75);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.37);
        transition: all 0.25s ease-in-out;
    }

    div[data-testid="stMetric"]:hover {
        border-color: rgba(6, 182, 212, 0.6);
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(6, 182, 212, 0.15);
    }

    div[data-testid="stMetricLabel"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: #94a3b8 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    div[data-testid="stMetricValue"] {
        font-family: 'Orbitron', sans-serif;
        font-weight: 700;
        color: #38bdf8 !important;
        text-shadow: 0 0 12px rgba(56, 189, 248, 0.4);
    }

    .status-card {
        background: rgba(15, 23, 42, 0.8);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .status-implemented { color: #10b981; font-weight: 600; text-shadow: 0 0 8px rgba(16, 185, 129, 0.4); }
    .status-planned { color: #f59e0b; font-weight: 600; }

    section[data-testid="stSidebar"] {
        background-color: #0a0f18 !important;
        border-right: 1px solid rgba(56, 189, 248, 0.12);
    }

    .stButton > button {
        border-radius: 10px;
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
        color: #f8fafc;
        border: 1px solid rgba(56, 189, 248, 0.4);
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        border-color: #38bdf8;
        box-shadow: 0 0 15px rgba(56, 189, 248, 0.4);
        transform: translateY(-1px);
    }
</style>
"""


def apply_theme() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
