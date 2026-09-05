"""
FastAPI Backend Application Entry Point.
Provides Swagger /docs, WebSocket /ws/live, and REST APIs for detection, tracking, behavior, anomaly, and reports.
"""

import os
import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import APIKeyHeader
from fastapi import HTTPException, Security, Depends

from utils.logger import get_logger
from utils.config_loader import load_config
from database.database import init_db
from api.routes import (
    detection_router,
    tracking_router,
    behavior_router,
    anomaly_router,
    history_router,
    reports_router,
    upload_router,
    live_router,
    health_router,
)

logger = get_logger(__name__)


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(api_key_header: str = Security(api_key_header)):
    cfg = load_config()
    expected_key = getattr(cfg.api, "api_key", "wl_super_secret_key_123")
    if api_key_header != expected_key:
        raise HTTPException(status_code=403, detail="Could not validate credentials")

app = FastAPI(
    title="AI Wildlife Animal Monitoring System API",
    description="Autonomous computer vision backend",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request performance logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = (time.perf_counter() - start_time) * 1000.0
    response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"
    return response

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled API exception on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "detail": str(exc)},
    )

# Mount all route routers with API Key security
app.include_router(detection_router, dependencies=[Depends(verify_api_key)])
app.include_router(tracking_router, dependencies=[Depends(verify_api_key)])
app.include_router(behavior_router, dependencies=[Depends(verify_api_key)])
app.include_router(anomaly_router, dependencies=[Depends(verify_api_key)])
app.include_router(history_router, dependencies=[Depends(verify_api_key)])
app.include_router(reports_router, dependencies=[Depends(verify_api_key)])
app.include_router(upload_router, dependencies=[Depends(verify_api_key)])
app.include_router(live_router, dependencies=[Depends(verify_api_key)])
app.include_router(health_router, dependencies=[Depends(verify_api_key)])

@app.get("/")
async def serve_dashboard():
    return {
        "status": "online",
        "service": "Nextra Wildlife Monitoring API",
        "docs_url": "/docs"
    }

@app.on_event("startup")
async def on_startup():
    logger.info("Starting up Wildlife Monitoring API backend...")
    try:
        init_db()
    except Exception as e:
        logger.warning(f"Database startup check note: {e}")
    logger.info("API initialized. Interactive Swagger docs available at /docs")

if __name__ == "__main__":
    import uvicorn
    cfg = load_config()
    uvicorn.run(app, host=cfg.api.host, port=cfg.api.port)
