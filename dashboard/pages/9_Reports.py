"""
Streamlit Page: Automated Census & Health Reports.
Allows generating and downloading Daily, Weekly, and Monthly reports in PDF and CSV formats.
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd

from dashboard.theme import apply_theme
from database.database import session_scope
from database.models import Report
from reports.report_generator import ReportGenerator

st.set_page_config(page_title="Automated Reports", page_icon="📑", layout="wide")
apply_theme()

st.title("📑 Wildlife Census & Health Report Generator")
st.markdown("Generate comprehensive reports compiling animal population counts, behavior analytics, health metrics, and perimeter security events.")

report_gen = ReportGenerator()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("⚙️ Generate New Report")
    report_type = st.selectbox("Report Cadence", ["daily", "weekly", "monthly"], format_func=lambda x: f"{x.capitalize()} Report")
    report_format = st.radio("Export Format", ["pdf", "csv"], format_func=lambda x: x.upper(), horizontal=True)

    if st.button("🚀 Compile Report Now", use_container_width=True):
        with st.spinner(f"Compiling {report_type} {report_format.upper()} report..."):
            try:
                path = report_gen.generate_report(report_type=report_type, report_format=report_format)
                st.success(f"Successfully compiled report: `{os.path.basename(path)}`")
            except Exception as e:
                st.error(f"Error compiling report: {e}")

with col2:
    st.subheader("📂 Available Generated Reports")
    with session_scope() as db:
        reports = db.query(Report).order_by(Report.generated_at.desc()).limit(15).all()
        if reports:
            for rep in reports:
                with st.expander(f"📄 {rep.report_type.upper()} ({rep.report_format.upper()}) - {rep.generated_at.strftime('%Y-%m-%d %H:%M')}"):
                    st.write(f"**File Path:** `{rep.file_path}`")
                    st.write(f"**Period:** {rep.period_start.strftime('%Y-%m-%d')} to {rep.period_end.strftime('%Y-%m-%d')}")
                    st.write(f"**Summary:** {rep.summary}")

                    if os.path.exists(rep.file_path):
                        with open(rep.file_path, "rb") as f:
                            btn_data = f.read()
                        mime = "application/pdf" if rep.report_format.lower() == "pdf" else "text/csv"
                        st.download_button(
                            label=f"⬇️ Download {os.path.basename(rep.file_path)}",
                            data=btn_data,
                            file_name=os.path.basename(rep.file_path),
                            mime=mime,
                            key=f"dl_{rep.id}",
                        )
                    else:
                        st.warning("File moved or archived.")
        else:
            st.info("No reports compiled yet. Use the generator on the left to produce one.")
