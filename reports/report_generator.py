"""
Unified Report Generation Coordinator for Wildlife Monitoring System.
"""

from typing import Dict, Any, Optional
import os
import datetime as dt

from utils.logger import get_logger
from database.database import session_scope
from database.models import Report, ReportType, ReportFormat
from reports.daily_report import generate_daily_report
from reports.weekly_report import generate_weekly_report, generate_monthly_report
from reports.pdf_report import PDFReportGenerator
from reports.csv_report import CSVReportGenerator

logger = get_logger(__name__)


class ReportGenerator:
    """
    Coordinates automatic generation and database registration of wildlife reports.
    """
    def __init__(self, output_dir: str = "outputs/reports"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_report(
        self,
        report_type: str = "daily",
        report_format: str = "pdf",
    ) -> str:
        rt = report_type.lower()
        rf = report_format.lower()
        now = dt.datetime.now(dt.timezone.utc)

        if rt == "weekly":
            period_start = now - dt.timedelta(days=7)
            file_path = generate_weekly_report(self.output_dir, fmt=rf)
            rep_type_enum = ReportType.WEEKLY
        elif rt == "monthly":
            period_start = now - dt.timedelta(days=30)
            file_path = generate_monthly_report(self.output_dir, fmt=rf)
            rep_type_enum = ReportType.MONTHLY
        else:
            period_start = now - dt.timedelta(days=1)
            file_path = generate_daily_report(self.output_dir, fmt=rf)
            rep_type_enum = ReportType.DAILY

        # Log record in database
        try:
            with session_scope() as db:
                rec = Report(
                    report_type=rt,
                    report_format=rf,
                    period_start=period_start,
                    period_end=now,
                    file_path=file_path,
                    summary=f"Automated {rt} {rf.upper()} report",
                )
                db.add(rec)
            logger.info(f"Report recorded in database: {file_path}")
        except Exception as e:
            logger.warning(f"Could not persist report metadata to database: {e}")

        return file_path
