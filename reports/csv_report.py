"""
CSV report generation utility using pandas.
"""

from typing import Dict, Any, List
import os
import pandas as pd
from utils.logger import get_logger

logger = get_logger(__name__)


class CSVReportGenerator:
    """
    Exports wildlife census, detections, and alerts to CSV files.
    """
    def generate(
        self,
        output_path: str,
        species_counts: Dict[str, int],
        alert_records: List[Dict[str, Any]],
        health_summary: Dict[str, Any],
    ) -> str:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        rows = []
        # Species summary
        for species, count in species_counts.items():
            rows.append({
                "Section": "SPECIES_CENSUS",
                "Key": species,
                "Value": count,
                "Details": "Total unique animals detected",
            })

        # Health metrics
        for k, v in health_summary.items():
            rows.append({
                "Section": "HEALTH_SUMMARY",
                "Key": k,
                "Value": v,
                "Details": "Aggregate kinematic health index",
            })

        # Alerts
        for a in alert_records:
            rows.append({
                "Section": "ALERTS",
                "Key": a.get("alert_type", "ALERT"),
                "Value": a.get("severity", "MEDIUM"),
                "Details": a.get("message", ""),
            })

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        logger.info(f"Generated CSV report: {output_path}")
        return output_path
