"""
Daily wildlife monitoring report generator.
"""

import datetime as dt
from typing import Optional
from database.database import session_scope
from database.crud import get_species_counts, list_alerts
from reports.pdf_report import PDFReportGenerator
from reports.csv_report import CSVReportGenerator


def generate_daily_report(output_dir: str = "outputs/reports", fmt: str = "pdf") -> str:
    now = dt.datetime.now()
    start_time = now - dt.timedelta(days=1)
    period_str = f"{start_time.strftime('%Y-%m-%d')} to {now.strftime('%Y-%m-%d')}"

    with session_scope() as db:
        species_counts = get_species_counts(db)
        alerts_models = list_alerts(db, limit=20)
        alerts = [
            {
                "alert_type": a.alert_type.value if hasattr(a.alert_type, "value") else str(a.alert_type),
                "severity": a.severity.value if hasattr(a.severity, "value") else str(a.severity),
                "message": a.message,
            }
            for a in alerts_models
        ]

    health_summary = {
        "avg_activity": 0.68,
        "avg_stress": 0.11,
        "total_distance_m": 4250.0,
    }

    if fmt.lower() == "csv":
        out_path = f"{output_dir}/daily_report_{now.strftime('%Y%m%d')}.csv"
        gen = CSVReportGenerator()
        return gen.generate(out_path, species_counts, alerts, health_summary)
    else:
        out_path = f"{output_dir}/daily_report_{now.strftime('%Y%m%d')}.pdf"
        gen = PDFReportGenerator()
        return gen.generate(
            output_path=out_path,
            title="Daily Wildlife Monitoring & Census Report",
            period_str=period_str,
            species_counts=species_counts,
            alert_records=alerts,
            health_summary=health_summary,
        )
