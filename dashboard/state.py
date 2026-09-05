"""
Cached resource accessors. Streamlit reruns the entire script top-to-bottom
on every widget interaction, so anything expensive (loading the YOLO model)
MUST be wrapped in st.cache_resource or the dashboard would reload the model
from disk on every click.
"""

import streamlit as st

from utils.config_loader import load_config
from detection.detector import get_detector as factory_get_detector, BaseDetector

AnimalDetector = BaseDetector


def get_config():
    """Config is cheap to (re)parse, so no caching needed here -- this also
    means editing configs/config.yaml takes effect on the next page rerun
    without needing to restart the whole dashboard."""
    return load_config()


@st.cache_resource(show_spinner="Loading detection model (first run only)...")
def get_detector() -> BaseDetector:
    cfg = load_config()
    return factory_get_detector(cfg)
