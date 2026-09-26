"""
FastAPI Backend Application Entry Point.
Compatible with both local full-featured deployment (YOLO, ByteTrack, PyTorch)
and Vercel Serverless Function deployment (lightweight DB, history, reports, health).
"""

import os
import time
from fastapi import FastAPI, Request, HTTPException, Security, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

from utils.logger import get_logger
from utils.config_loader import load_config
from database.database import init_db

# Core / Lightweight routers (Safe for Vercel serverless functions)
from api.routes.health import router as health_router
from api.routes.history import router as history_router
from api.routes.reports import router as reports_router
from api.routes.upload import router as upload_router
from api.routes.live import router as live_router

logger = get_logger(__name__)

# Authentication setup
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str = Security(api_key_header)):
    """Validates API key if configured; allows open development access when default/unset."""
    cfg = load_config()
    expected_key = getattr(cfg.api, "api_key", None) or os.getenv("API_SECRET_KEY")
    # Enforce check only when a non-default custom secret key is configured
    if expected_key and expected_key not in ("wl_super_secret_key_123", "change_this_in_production_secret_key_wildlife_2026"):
        if api_key != expected_key:
            raise HTTPException(status_code=403, detail="Could not validate credentials")
    return True


app = FastAPI(
    title="AI Wildlife Animal Monitoring System API",
    description="Autonomous computer vision and wildlife reserve database backend",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
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


# 1. Mount always-available serverless / database routes
app.include_router(health_router)
app.include_router(history_router, dependencies=[Depends(verify_api_key)])
app.include_router(reports_router, dependencies=[Depends(verify_api_key)])
app.include_router(upload_router, dependencies=[Depends(verify_api_key)])
app.include_router(live_router, dependencies=[Depends(verify_api_key)])

# 2. Conditionally import & mount heavy AI/ML routers (OpenCV, PyTorch, Ultralytics)
ml_features_loaded = {}

try:
    from api.routes.detection import router as detection_router
    app.include_router(detection_router, dependencies=[Depends(verify_api_key)])
    ml_features_loaded["detection"] = True
except Exception as e:
    logger.info(f"Detection router disabled (ML packages not loaded): {e}")
    ml_features_loaded["detection"] = False

try:
    from api.routes.tracking import router as tracking_router
    app.include_router(tracking_router, dependencies=[Depends(verify_api_key)])
    ml_features_loaded["tracking"] = True
except Exception as e:
    logger.info(f"Tracking router disabled (ML packages not loaded): {e}")
    ml_features_loaded["tracking"] = False

try:
    from api.routes.behavior import router as behavior_router
    app.include_router(behavior_router, dependencies=[Depends(verify_api_key)])
    ml_features_loaded["behavior"] = True
except Exception as e:
    logger.info(f"Behavior router disabled (ML packages not loaded): {e}")
    ml_features_loaded["behavior"] = False

try:
    from api.routes.anomaly import router as anomaly_router
    app.include_router(anomaly_router, dependencies=[Depends(verify_api_key)])
    ml_features_loaded["anomaly"] = True
except Exception as e:
    logger.info(f"Anomaly router disabled (ML packages not loaded): {e}")
    ml_features_loaded["anomaly"] = False


@app.get("/")
async def serve_root():
    is_serverless = any(not loaded for loaded in ml_features_loaded.values())
    return {
        "status": "online",
        "service": "AI Wildlife Animal Monitoring API",
        "deployment": "Serverless (Vercel)" if is_serverless else "Full Engine (Local/GPU)",
        "features": {
            "database_api": True,
            "reports": True,
            "health": True,
            **ml_features_loaded,
        },
        "docs_url": "/docs",
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
