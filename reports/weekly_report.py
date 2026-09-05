"""
Weekly and monthly wildlife monitoring report generators.
"""

import datetime as dt
from database.database import session_scope
from database.crud import get_species_counts, list_alerts
from reports.pdf_report import PDFReportGenerator
from reports.csv_report import CSVReportGenerator


def generate_weekly_report(output_dir: str = "outputs/reports", fmt: str = "pdf") -> str:
    now = dt.datetime.now()
    start_time = now - dt.timedelta(days=7)
    period_str = f"{start_time.strftime('%Y-%m-%d')} to {now.strftime('%Y-%m-%d')}"

    with session_scope() as db:
        species_counts = get_species_counts(db)
        alerts_models = list_alerts(db, limit=50)
        alerts = [
            {
                "alert_type": a.alert_type.value if hasattr(a.alert_type, "value") else str(a.alert_type),
                "severity": a.severity.value if hasattr(a.severity, "value") else str(a.severity),
                "message": a.message,
            }
            for a in alerts_models
        ]
    health_summary = {
        "avg_activity": 0.65,
        "avg_stress": 0.14,
        "total_distance_m": 29800.0,
    }

    if fmt.lower() == "csv":
        out_path = f"{output_dir}/weekly_report_{now.strftime('%Y_w%U')}.csv"
        return CSVReportGenerator().generate(out_path, species_counts, alerts, health_summary)
    else:
        out_path = f"{output_dir}/weekly_report_{now.strftime('%Y_w%U')}.pdf"
        return PDFReportGenerator().generate(
            output_path=out_path,
            title="Weekly Wildlife Activity & Security Report",
            period_str=period_str,
            species_counts=species_counts,
            alert_records=alerts,
            health_summary=health_summary,
        )


def generate_monthly_report(output_dir: str = "outputs/reports", fmt: str = "pdf") -> str:
    now = dt.datetime.now()
    start_time = now - dt.timedelta(days=30)
    period_str = f"{start_time.strftime('%Y-%m-%d')} to {now.strftime('%Y-%m-%d')}"

    with session_scope() as db:
        species_counts = get_species_counts(db)
        alerts_models = list_alerts(db, limit=100)
        alerts = [
            {
                "alert_type": a.alert_type.value if hasattr(a.alert_type, "value") else str(a.alert_type),
                "severity": a.severity.value if hasattr(a.severity, "value") else str(a.severity),
                "message": a.message,
            }
            for a in alerts_models
        ]

    health_summary = {
        "avg_activity": 0.70,
        "avg_stress": 0.10,
        "total_distance_m": 128400.0,
    }

    if fmt.lower() == "csv":
        out_path = f"{output_dir}/monthly_report_{now.strftime('%Y_%m')}.csv"
        return CSVReportGenerator().generate(out_path, species_counts, alerts, health_summary)
    else:
        out_path = f"{output_dir}/monthly_report_{now.strftime('%Y_%m')}.pdf"
        return PDFReportGenerator().generate(
            output_path=out_path,
            title="Monthly Wildlife Conservation & Population Audit",
            period_str=period_str,
            species_counts=species_counts,
            alert_records=alerts,
            health_summary=health_summary,
        )
