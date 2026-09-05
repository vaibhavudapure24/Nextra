"""
Security utilities:
- Uploaded file validation (size, MIME type, extension)
- Path traversal protection and filename sanitization
- API token validation
"""

import os
import re
import secrets
from typing import Tuple, Set
from fastapi import HTTPException, status
from utils.logger import get_logger

logger = get_logger(__name__)

# Allowed file extensions for media uploads
ALLOWED_IMAGE_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_VIDEO_EXTENSIONS: Set[str] = {".mp4", ".avi", ".mov", ".mkv"}
ALLOWED_EXTENSIONS: Set[str] = ALLOWED_IMAGE_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS

# File size limits (in bytes)
MAX_IMAGE_SIZE_BYTES = 25 * 1024 * 1024       # 25 MB
MAX_VIDEO_SIZE_BYTES = 150 * 1024 * 1024      # 150 MB

# Secret key loaded from environment
API_SECRET_KEY = os.getenv("API_SECRET_KEY", "wildlife_secure_token_2026_phase1")


def sanitize_filename(filename: str) -> str:
    """
    Sanitizes an input filename to prevent directory traversal and illegal characters.
    """
    # Extract basename only
    base = os.path.basename(filename)
    # Remove any characters except alphanumeric, dashes, dots, underscores
    sanitized = re.sub(r"[^a-zA-Z0-9_.-]", "_", base)
    # Ensure not empty
    if not sanitized or sanitized.startswith("."):
        sanitized = f"upload_{secrets.token_hex(4)}_{sanitized}"
    return sanitized


def validate_file_upload(
    filename: str, content_size: int, is_video: bool = False
) -> Tuple[bool, str]:
    """
    Validates uploaded file size and extension.
    """
    ext = os.path.splitext(filename)[1].lower()
    allowed = ALLOWED_VIDEO_EXTENSIONS if is_video else ALLOWED_EXTENSIONS
    max_size = MAX_VIDEO_SIZE_BYTES if is_video else MAX_IMAGE_SIZE_BYTES

    if ext not in allowed:
        return False, f"Unsupported file extension '{ext}'. Allowed: {sorted(list(allowed))}"

    if content_size > max_size:
        return False, f"File size ({content_size / (1024*1024):.1f} MB) exceeds maximum allowed limit ({max_size / (1024*1024):.1f} MB)"

    return True, "Valid"


def verify_api_token(token: str) -> bool:
    """
    Verifies authentication token using timing-attack resistant comparison.
    """
    if not token:
        return False
    return secrets.compare_digest(token, API_SECRET_KEY)
