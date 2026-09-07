"""
Vercel Database API Entrypoint.
This file defines a lightweight FastAPI app that only imports database routes (history, reports, health).
It completely avoids importing cv2, torch, or ultralytics to satisfy Vercel's 250MB limit.
"""

from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time

from utils.logger import get_logger
from database.database import init_db

# Only import routes that do NOT use OpenCV or PyTorch
from api.routes.history import router as history_router
from api.routes.reports import router as reports_router
from api.routes.health import router as health_router

logger = get_logger(__name__)

app = FastAPI(
    title="Wildlife Database API (Vercel Serverless)",
    description="Lightweight Database API. AI inference runs locally.",
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

# Mount only the database-safe routers
app.include_router(history_router)
app.include_router(reports_router)
app.include_router(health_router)

@app.get("/")
async def serve_status():
    return {
        "status": "online",
        "service": "Nextra Wildlife Database API",
        "mode": "Serverless (AI stripped)",
        "docs_url": "/docs"
    }

@app.on_event("startup")
async def on_startup():
    logger.info("Starting up Vercel Database API...")
    try:
        init_db()
    except Exception as e:
        logger.warning(f"Database startup check note: {e}")
