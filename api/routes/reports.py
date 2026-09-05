"""
Reports API Router.
Provides GET /report (listing/downloading reports) and POST /report/generate (compiling new reports).
"""

from typing import List, Optional
import os
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from database.database import get_db
from database.models import Report
from reports.report_generator import ReportGenerator

router = APIRouter(prefix="", tags=["Reports"])

_report_gen = ReportGenerator()


@router.get("/report")
def list_or_download_reports(
    report_id: Optional[int] = Query(None),
    download: bool = Query(False),
    db: Session = Depends(get_db),
):
    """
    Lists generated reports or downloads a specific PDF/CSV report file.
    """
    if report_id is not None:
        rep = db.query(Report).filter(Report.id == report_id).first()
        if not rep:
            raise HTTPException(status_code=404, detail=f"Report ID {report_id} not found.")

        if download:
            if not os.path.exists(rep.file_path):
                raise HTTPException(status_code=404, detail="Report file missing on filesystem.")
            media_type = "application/pdf" if rep.report_format.lower() == "pdf" else "text/csv"
            return FileResponse(rep.file_path, media_type=media_type, filename=os.path.basename(rep.file_path))

        return {
            "id": rep.id,
            "report_type": rep.report_type,
            "report_format": rep.report_format,
            "period_start": rep.period_start.isoformat(),
            "period_end": rep.period_end.isoformat(),
            "file_path": rep.file_path,
            "summary": rep.summary,
            "generated_at": rep.generated_at.isoformat(),
        }

    reports = db.query(Report).order_by(Report.generated_at.desc()).limit(20).all()
    return [
        {
            "id": r.id,
            "report_type": r.report_type,
            "report_format": r.report_format,
            "file_path": r.file_path,
            "summary": r.summary,
            "generated_at": r.generated_at.isoformat(),
        }
        for r in reports
    ]


@router.post("/report/generate")
def generate_report_endpoint(
    report_type: str = Query("daily", pattern="^(daily|weekly|monthly)$"),
    report_format: str = Query("pdf", pattern="^(pdf|csv)$"),
):
    """
    Generates a new wildlife activity report and records it in the database.
    """
    try:
        path = _report_gen.generate_report(report_type=report_type, report_format=report_format)
        return {
            "status": "success",
            "report_type": report_type,
            "report_format": report_format,
            "file_path": path,
            "filename": os.path.basename(path),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation error: {e}")
