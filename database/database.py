"""
Database engine and session management.
Connects to PostgreSQL by default, with automatic SQLite fallback if PostgreSQL is offline,
guaranteeing high availability and preventing crashes.
"""

from __future__ import annotations
import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import OperationalError

from utils.config_loader import load_config
from utils.logger import get_logger
from database.models import Base

logger = get_logger(__name__)

_cfg = load_config()
_db_cfg = _cfg.database

# Check if explicit DATABASE_URL is set in environment
ENV_DB_URL = os.getenv("DATABASE_URL")
if ENV_DB_URL:
    DEFAULT_DATABASE_URL = ENV_DB_URL
else:
    DEFAULT_DATABASE_URL = (
        f"postgresql+psycopg2://{_db_cfg.user}:{_db_cfg.password}"
        f"@{_db_cfg.host}:{_db_cfg.port}/{_db_cfg.name}"
    )

SQLITE_FALLBACK_URL = "sqlite:///wildlife_monitoring.db"

# Create engine with fallback logic
engine = None
try:
    if "sqlite" in DEFAULT_DATABASE_URL:
        engine = create_engine(DEFAULT_DATABASE_URL, connect_args={"check_same_thread": False})
        logger.info(f"Using SQLite database: {DEFAULT_DATABASE_URL}")
    else:
        # Try connecting to PostgreSQL
        test_engine = create_engine(
            DEFAULT_DATABASE_URL,
            pool_size=_db_cfg.pool_size,
            pool_pre_ping=True,
            echo=_db_cfg.echo,
            connect_args={"connect_timeout": 3},
        )
        # Test connection
        with test_engine.connect() as conn:
            pass
        engine = test_engine
        logger.info(f"Successfully connected to PostgreSQL at {_db_cfg.host}:{_db_cfg.port}/{_db_cfg.name}")
except Exception as e:
    logger.warning(
        f"PostgreSQL connection to '{DEFAULT_DATABASE_URL}' failed: {e}. "
        f"Gracefully falling back to local SQLite database: {SQLITE_FALLBACK_URL}"
    )
    engine = create_engine(SQLITE_FALLBACK_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, expire_on_commit=False)


def init_db() -> None:
    """Create all tables if they do not already exist."""
    logger.info("Initializing database schema...")
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database schema initialized successfully.")
    except Exception as e:
        logger.error(f"Error during schema initialization: {e}")
        raise


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency: yields a session and guarantees cleanup."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """Context manager for background pipelines and CLI scripts."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    init_db()
    print("Database initialization complete.")
