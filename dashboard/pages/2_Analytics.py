"""Historical analytics, pulled live from PostgreSQL. Gracefully degrades
with a clear message if the database isn't reachable or has no data yet."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.theme import apply_theme

st.set_page_config(page_title="Analytics - AI Wildlife Monitoring", page_icon="📊", layout="wide")
apply_theme()

st.title("📊 Detection Analytics")
st.caption(
    "Historical data pulled live from PostgreSQL. Requires the database to be "
    "running and initialized (`python -m database.init_db`)."
)

if st.button("🔄 Refresh"):
    st.rerun()

recent = []
animals = []
db_error = None
try:
    from database.database import SessionLocal
    from database.crud import get_detection_history, list_animals

    db = SessionLocal()
    try:
        recent = get_detection_history(db, limit=2000)
        animals = list_animals(db)
    finally:
        db.close()
except Exception as exc:
    db_error = exc

if db_error is not None:
    st.error(f"Could not connect to the database: {db_error}")
    st.info(
        "Start PostgreSQL (see docker-compose.yml) and run `python -m database.init_db`, "
        "or check configs/config.yaml -> database.* for the correct host/port/credentials."
    )
elif not recent:
    st.info(
        "No detections logged yet. Run `python main.py` without `--no-db`, or enable "
        "'Log detections to database' on the Live Feed page, to start collecting data."
    )
else:
    df = pd.DataFrame([
        {
            "timestamp": d.timestamp,
            "species": d.species,
            "confidence": d.confidence,
            "camera_id": d.camera_id or "unknown",
        }
        for d in recent
    ])

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Detections", len(df))
    col2.metric("Unique Species", df["species"].nunique())
    col3.metric("Tracked Animal Records", len(animals))

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        species_counts = df["species"].value_counts().reset_index()
        species_counts.columns = ["species", "count"]
        fig1 = px.bar(
            species_counts, x="species", y="count", color="species",
            title="Detections by Species", template="plotly_dark",
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col_b:
        fig_conf = px.histogram(
            df, x="confidence", nbins=20, title="Confidence Score Distribution",
            template="plotly_dark",
        )
        st.plotly_chart(fig_conf, use_container_width=True)

    df["hour"] = pd.to_datetime(df["timestamp"]).dt.floor("h")
    timeline = df.groupby("hour").size().reset_index(name="count")
    fig2 = px.line(
        timeline, x="hour", y="count", markers=True,
        title="Detections Over Time", template="plotly_dark",
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Recent Detections")
    st.dataframe(
        df.sort_values("timestamp", ascending=False).head(200),
        use_container_width=True,
    )
