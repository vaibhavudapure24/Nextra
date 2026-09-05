"""
Generates synthetic sample media so the pipeline (main.py) has something to
run against immediately, without needing your own footage first.

IMPORTANT: this creates synthetic shapes, NOT real animal photos/video.
It's a "plumbing test" only -- it proves the video/image reading pipeline,
detection loop, and logging all work end-to-end. Since there is no real
animal in these frames, YOLO will correctly detect 0 animals in them. That
is expected and does NOT indicate a bug. To actually test detection
accuracy, point configs/config.yaml at your own real photo/video, or use
--source usb for a live camera.

Usage:
    python -m datasets.generate_test_media
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from utils.config_loader import resolve_path
from utils.logger import get_logger

logger = get_logger(__name__)

WIDTH, HEIGHT = 640, 480
FPS = 30
DURATION_SECONDS = 5


def generate_frame(t: float) -> np.ndarray:
    """A moving colored blob on a gradient background -- enough motion/content
    to exercise the full read -> detect -> draw -> display/save loop."""
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    for y in range(HEIGHT):
        frame[y, :] = (40, 60 + int(40 * y / HEIGHT), 30)

    cx = int(WIDTH / 2 + (WIDTH / 3) * math.sin(t))
    cy = int(HEIGHT / 2 + (HEIGHT / 4) * math.cos(t * 0.7))
    cv2.circle(frame, (cx, cy), 40, (60, 140, 220), -1)
    cv2.circle(frame, (cx, cy), 40, (255, 255, 255), 2)

    cv2.putText(
        frame, "SYNTHETIC TEST FRAME (no real animal)", (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
    )
    return frame


def generate_video(output_path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    total_frames = FPS * DURATION_SECONDS
    for i in range(total_frames):
        t = i / FPS
        writer.write(generate_frame(t))
    writer.release()
    logger.info(f"Wrote {total_frames} frames ({DURATION_SECONDS}s @ {FPS}fps) to {output_path}")


def generate_images(output_dir, count: int = 10) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        t = i / 3.0
        frame = generate_frame(t)
        path = output_dir / f"synthetic_{i:03d}.jpg"
        cv2.imwrite(str(path), frame)
    logger.info(f"Wrote {count} synthetic images to {output_dir}")


def main() -> None:
    video_path = resolve_path("datasets/sample_videos/sample.mp4")
    images_dir = resolve_path("datasets/sample_images")

    generate_video(video_path)
    generate_images(images_dir)

    logger.info("=" * 70)
    logger.info("Synthetic test media generated. Try:")
    logger.info("  python main.py --source video_file --no-db")
    logger.info("  python main.py --source image_folder --no-db")
    logger.info("These frames contain NO real animal, so 0 detections is the")
    logger.info("CORRECT result -- this only proves the pipeline itself runs.")
    logger.info("For real detection testing, use your own photo/video or --source usb.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
