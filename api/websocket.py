"""
WebSocket manager for live real-time surveillance streaming.
Broadcasts frame telemetry, detections, tracking, behaviors, anomalies, and alerts.
"""

from typing import List, Dict, Any
from fastapi import WebSocket, WebSocketDisconnect
from utils.logger import get_logger

logger = get_logger(__name__)


class ConnectionManager:
    """
    Manages active WebSocket client connections and broadcasts live updates.
    """
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected. Active: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"WebSocket client disconnected. Active: {len(self.active_connections)}")

    async def broadcast(self, message: Dict[str, Any]):
        """
        Broadcasts live JSON payload to all connected clients (e.g. Streamlit dashboards).
        """
        dead_connections = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                dead_connections.append(connection)

        for dead in dead_connections:
            self.disconnect(dead)


ws_manager = ConnectionManager()
