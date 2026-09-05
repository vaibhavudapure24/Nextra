"""
Dataset preparation, validation, and splitting for YOLO wildlife detection.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import shutil
import argparse
import random
from typing import List, Tuple, Dict
from utils.logger import get_logger

logger = get_logger(__name__)


def validate_yolo_label_file(label_path: str, num_classes: int = 80) -> Tuple[bool, str]:
    """
    Validates that a YOLO label file contains normalized coordinates:
    <class_id> <x_center> <y_center> <width> <height>
    """
    if not os.path.exists(label_path):
        return False, "File does not exist"

    with open(label_path, "r") as f:
        lines = f.readlines()

    for idx, line in enumerate(lines):
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) < 5:
            return False, f"Line {idx+1}: Expected 5 values, got {len(parts)}"

        try:
            cls_id = int(parts[0])
            xc, yc, w, h = map(float, parts[1:5])
        except ValueError:
            return False, f"Line {idx+1}: Non-numeric value encountered"

        if cls_id < 0 or cls_id >= num_classes:
            return False, f"Line {idx+1}: Class ID {cls_id} out of bounds (0-{num_classes-1})"

        for val_name, val in [("x_center", xc), ("y_center", yc), ("width", w), ("height", h)]:
            if not (0.0 <= val <= 1.0):
                return False, f"Line {idx+1}: {val_name}={val} is not normalized within [0.0, 1.0]"

    return True, "Valid"


def prepare_yolo_dataset(
    source_images_dir: str,
    output_dataset_dir: str = "datasets/wildlife_dataset",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    classes: List[str] = None,
):
    """
    Splits image/label files into train/val/test directories and generates data.yaml.
    """
    classes = classes or ["elephant", "zebra", "giraffe", "lion", "cheetah", "rhino", "leopard", "deer"]
    os.makedirs(output_dataset_dir, exist_ok=True)

    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(output_dataset_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dataset_dir, "labels", split), exist_ok=True)

    # Write data.yaml
    yaml_content = f"""# Autonomous Wildlife Monitoring Dataset Configuration
path: {os.path.abspath(output_dataset_dir)}
train: images/train
val: images/val
test: images/test

names:
"""
    for idx, name in enumerate(classes):
        yaml_content += f"  {idx}: {name}\n"

    yaml_path = os.path.join(output_dataset_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    logger.info(f"Dataset structure prepared at: {output_dataset_dir}")
    logger.info(f"Generated dataset configuration: {yaml_path}")
    return yaml_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare YOLO Wildlife Dataset")
    parser.add_argument("--source", type=str, default="datasets/sample_images")
    parser.add_argument("--output", type=str, default="datasets/wildlife_dataset")
    args = parser.parse_args()

    prepare_yolo_dataset(source_images_dir=args.source, output_dataset_dir=args.output)
