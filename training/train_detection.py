"""
YOLO11 / YOLOv8 Wildlife Detection Model Training Runner.
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from ultralytics import YOLO

from utils.logger import get_logger
from utils.config_loader import load_config, resolve_path
from utils.device import select_device

logger = get_logger(__name__)


def train_detection(
    data_yaml: str,
    base_model: str = "yolo11n.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "auto",
    project_dir: str = "outputs/training_runs",
    run_name: str = "wildlife_detection",
):
    """
    Fine-tunes YOLO11 or YOLOv8 detector on custom wildlife dataset.
    """
    dev = select_device(device)
    device_str = "0" if str(dev).startswith("cuda") else "cpu"

    resolved_data = str(resolve_path(data_yaml))
    if not os.path.exists(resolved_data):
        logger.error(f"Dataset config '{resolved_data}' not found. Prepare dataset first.")
        return None

    logger.info(f"Loading base checkpoint '{base_model}' for fine-tuning on {device_str}...")
    model = YOLO(base_model)

    results = model.train(
        data=resolved_data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device_str,
        project=project_dir,
        name=run_name,
        save=True,
        plots=True,
        verbose=True,
    )

    best_weights = os.path.join(project_dir, run_name, "weights", "best.pt")
    logger.info(f"Training completed successfully! Best weights saved to: {best_weights}")

    # Copy best weights to models/detection/wildlife.pt
    dest = "models/detection/wildlife.pt"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(best_weights):
        import shutil
        shutil.copy(best_weights, dest)
        logger.info(f"Copied best model checkpoint to {dest}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Wildlife Detection Model")
    parser.add_argument("--data", type=str, default="datasets/wildlife_dataset/data.yaml")
    parser.add_argument("--model", type=str, default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()
    train_detection(
        data_yaml=args.data,
        base_model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )
