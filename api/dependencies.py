"""
FastAPI Dependencies for database sessions, authentication, and configurations.
"""

from typing import Generator, Optional
from fastapi import Depends, HTTPException, Header, status
from sqlalchemy.orm import Session

from database.database import get_db
from utils.config_loader import load_config, AppConfig
from utils.security import verify_api_token


def get_current_config() -> AppConfig:
    return load_config()


def optional_auth(x_api_token: Optional[str] = Header(None)) -> bool:
    """
    Optional API token verification header for security-sensitive endpoints.
    Allows unauthenticated requests in development, but validates if provided.
    """
    if x_api_token and not verify_api_token(x_api_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired API token",
        )
    return True
