"""
Upload API Router.
Provides POST /upload for secure media file ingestion (camera traps, drone flights, field recordings).
"""

import os
import shutil
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from utils.logger import get_logger
from utils.security import sanitize_filename, validate_file_upload

logger = get_logger(__name__)

router = APIRouter(prefix="", tags=["Upload"])


@router.post("/upload")
async def upload_media_file(
    file: UploadFile = File(...),
    camera_id: str = Form("upload_source"),
    description: str = Form(""),
):
    """
    Securely uploads an image or video file with size limits and path traversal protection.
    """
    sanitized_name = sanitize_filename(file.filename or "uploaded_media")
    is_video = sanitized_name.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))

    # Determine destination folder
    dest_dir = "outputs/videos" if is_video else "outputs/snapshots"
    try:
        os.makedirs(dest_dir, exist_ok=True)
    except OSError:
        dest_dir = os.path.join("/tmp", dest_dir)
        os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, sanitized_name)

    # Read and validate size in chunks
    file_size = 0
    with open(dest_path, "wb") as buffer:
        while chunk := await file.read(1024 * 1024):  # 1MB chunks
            file_size += len(chunk)
            # Validate size limits
            valid, msg = validate_file_upload(sanitized_name, file_size, is_video=is_video)
            if not valid:
                buffer.close()
                if os.path.exists(dest_path):
                    os.remove(dest_path)
                raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=msg)
            buffer.write(chunk)

    logger.info(f"Successfully uploaded {sanitized_name} ({file_size / (1024*1024):.2f} MB)")

    return {
        "status": "success",
        "filename": sanitized_name,
        "file_path": dest_path,
        "size_bytes": file_size,
        "media_type": "video" if is_video else "image",
        "camera_id": camera_id,
    }
