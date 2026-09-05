"""
Centralized logging setup for the AI Wildlife Monitoring System.

Every module should do:
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("message")

Logs are written both to stdout (colorized) and to rotating files under
the configured logs/ directory, split by day.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from loguru import logger as _loguru_logger

from utils.config_loader import load_config, resolve_path

_CONFIGURED = False


def _configure_once() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    try:
        cfg = load_config()
        level = cfg.logging.level
        log_dir = resolve_path(cfg.logging.log_dir)
        rotation = cfg.logging.rotation
        retention = cfg.logging.retention
    except Exception:
        # Fall back to sane defaults if config isn't available yet
        # (e.g. during very early bootstrap or isolated unit tests).
        level = "INFO"
        log_dir = Path(__file__).resolve().parent.parent / "logs"
        rotation = "10 MB"
        retention = "30 days"

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    _loguru_logger.remove()  # remove default handler to avoid duplicate stdout logs

    _loguru_logger.add(
        sys.stdout,
        level=level,
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
    )

    _loguru_logger.add(
        str(log_dir / "wildlife_system_{time:YYYY-MM-DD}.log"),
        level=level,
        rotation=rotation,
        retention=retention,
        encoding="utf-8",
        enqueue=True,       # thread/process-safe
        backtrace=True,
        diagnose=False,
    )

    _loguru_logger.add(
        str(log_dir / "errors_{time:YYYY-MM-DD}.log"),
        level="ERROR",
        rotation=rotation,
        retention=retention,
        encoding="utf-8",
        enqueue=True,
    )

    _CONFIGURED = True


def get_logger(name: Optional[str] = None):
    """Return a loguru logger bound with the calling module's name."""
    _configure_once()
    return _loguru_logger.bind(module=name or "wildlife_system")


if __name__ == "__main__":
    log = get_logger(__name__)
    log.info("Logger initialized successfully.")
    log.warning("This is a warning-level test message.")
    log.error("This is an error-level test message.")
