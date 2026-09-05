"""
PDF Report Generator using ReportLab.
Renders census tables, alert summaries, and health metrics into formatted PDF documents.
"""

from typing import Dict, Any, List
import os
import datetime as dt

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from utils.logger import get_logger

logger = get_logger(__name__)


class PDFReportGenerator:
    """
    Renders wildlife surveillance reports into PDF format.
    """
    def generate(
        self,
        output_path: str,
        title: str,
        period_str: str,
        species_counts: Dict[str, int],
        alert_records: List[Dict[str, Any]],
        health_summary: Dict[str, Any],
        camera_uptime_pct: float = 99.4,
    ) -> str:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        styles = getSampleStyleSheet()

        elements = []

        # Title
        title_style = ParagraphStyle(
            "ReportTitle",
            parent=styles["Heading1"],
            fontSize=22,
            textColor=colors.HexColor("#2c3e50"),
            spaceAfter=12,
        )
        elements.append(Paragraph(title, title_style))

        # Metadata
        meta_style = ParagraphStyle(
            "MetaStyle",
            parent=styles["Normal"],
            fontSize=10,
            textColor=colors.HexColor("#7f8c8d"),
            spaceAfter=18,
        )
        elements.append(Paragraph(f"Reporting Period: {period_str} | Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}", meta_style))
        elements.append(Spacer(1, 10))

        # 1. Species Census Table
        elements.append(Paragraph("<b>1. Wildlife Species Census</b>", styles["Heading2"]))
        species_data = [["Species", "Count Observed", "Status"]]
        for sp, count in (species_counts.items() if species_counts else [("No observations", 0)]):
            species_data.append([sp, str(count), "Monitored"])

        t_species = Table(species_data, colWidths=[200, 150, 150])
        t_species.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#bdc3c7")),
        ]))
        elements.append(t_species)
        elements.append(Spacer(1, 15))

        # 2. Operational Health & Surveillance Statistics
        elements.append(Paragraph("<b>2. Health & Surveillance Statistics</b>", styles["Heading2"]))
        health_data = [
            ["Metric", "Value"],
            ["Camera Uptime", f"{camera_uptime_pct:.1f}%"],
            ["Average Activity Level", f"{health_summary.get('avg_activity', 0.65)*100:.1f}%"],
            ["Average Stress Index", f"{health_summary.get('avg_stress', 0.12)*100:.1f}%"],
            ["Total Walking Distance Recorded", f"{health_summary.get('total_distance_m', 0.0):.1f} m"],
        ]
        t_health = Table(health_data, colWidths=[250, 250])
        t_health.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2980b9")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#bdc3c7")),
        ]))
        elements.append(t_health)
        elements.append(Spacer(1, 15))

        # 3. Security & Safety Alerts Summary
        elements.append(Paragraph("<b>3. Critical Alerts & Security Events</b>", styles["Heading2"]))
        alert_data = [["Type", "Severity", "Message"]]
        if alert_records:
            for a in alert_records[:10]:
                alert_data.append([
                    str(a.get("alert_type", "ALERT")),
                    str(a.get("severity", "MEDIUM")),
                    str(a.get("message", ""))[:45] + "...",
                ])
        else:
            alert_data.append(["None", "INFO", "No security or medical alerts during period."])

        t_alert = Table(alert_data, colWidths=[120, 80, 300])
        t_alert.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e74c3c")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#bdc3c7")),
        ]))
        elements.append(t_alert)

        doc.build(elements)
        logger.info(f"Generated PDF report: {output_path}")
        return output_path
