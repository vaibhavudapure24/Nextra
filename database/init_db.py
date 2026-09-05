"""
Standalone script to initialize the PostgreSQL schema.

Usage:
    python -m database.init_db

Make sure PostgreSQL is running and configs/config.yaml (or the
DATABASE__* environment variables) point to a reachable instance first.
See docker-compose.yml for a ready-to-run local Postgres instance.
"""

from database.database import init_db
from utils.logger import get_logger

logger = get_logger(__name__)

if __name__ == "__main__":
    logger.info("Creating database tables (if they do not already exist)...")
    init_db()
    logger.info("Done.")
