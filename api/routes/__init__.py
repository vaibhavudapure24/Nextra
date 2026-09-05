"""
FastAPI route definitions.
"""

from api.routes.detection import router as detection_router
from api.routes.tracking import router as tracking_router
from api.routes.behavior import router as behavior_router
from api.routes.anomaly import router as anomaly_router
from api.routes.history import router as history_router
from api.routes.reports import router as reports_router
from api.routes.upload import router as upload_router
from api.routes.live import router as live_router
from api.routes.health import router as health_router

__all__ = [
    "detection_router",
    "tracking_router",
    "behavior_router",
    "anomaly_router",
    "history_router",
    "reports_router",
    "upload_router",
    "live_router",
    "health_router",
]
