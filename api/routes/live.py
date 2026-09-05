"""
Live Surveillance and WebSocket API Router.
Provides GET /live for system stream status and WebSocket /ws/live for real-time AI events.
"""

from typing import Dict, Any
import time
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session

from database.database import get_db
from database.crud import get_dashboard_counts, list_alerts
from api.websocket import ws_manager
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="", tags=["Live"])


@router.get("/live")
def get_live_status(db: Session = Depends(get_db)):
    """
    Returns real-time surveillance metrics: active cameras, recent alerts, and tracking counts.
    """
    counts = get_dashboard_counts(db)
    recent_alerts = list_alerts(db, unacknowledged_only=True, limit=5)

    return {
        "status": "online",
        "active_cameras": counts.get("total_cameras", 0),
        "total_tracked_animals": counts.get("total_animals", 0),
        "recent_alerts": [
            {
                "id": a.id,
                "type": a.alert_type.value if hasattr(a.alert_type, "value") else str(a.alert_type),
                "severity": a.severity.value if hasattr(a.severity, "value") else str(a.severity),
                "message": a.message,
                "camera_id": a.camera_id,
                "time": a.timestamp.isoformat(),
            }
            for a in recent_alerts
        ],
        "timestamp": time.time(),
    }


@router.websocket("/ws/live")
async def websocket_live_endpoint(websocket: WebSocket):
    """
    WebSocket channel broadcasting live detection, tracking, behavior, and alert events.
    """
    await ws_manager.connect(websocket)
    try:
        while True:
            # Client can ping or send control frames
            data = await websocket.receive_text()
            # Send acknowledgement with server timestamp
            await websocket.send_json({
                "type": "heartbeat",
                "client_msg": data,
                "server_time": time.time(),
            })
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        logger.warning(f"WebSocket error: {e}")
        ws_manager.disconnect(websocket)
