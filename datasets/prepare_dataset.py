"""
Prepares a flat folder of images + YOLO-format .txt labels into the
train/val/test directory structure Ultralytics expects, and generates the
data.yaml file that training/train_yolo.py and training/evaluate.py consume.

Usage:
    python -m datasets.prepare_dataset \\
        --images datasets/raw/images \\
        --labels datasets/raw/labels \\
        --classes datasets/raw/classes.txt \\
        --output datasets/wildlife_dataset

See datasets/README.md for the expected input layout and the YOLO label format.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import yaml

from utils.logger import get_logger
from utils.config_loader import load_config, resolve_path

logger = get_logger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a YOLO-format wildlife dataset")
    parser.add_argument("--images", required=True, help="Folder of raw images")
    parser.add_argument("--labels", required=True, help="Folder of matching YOLO .txt label files")
    parser.add_argument("--classes", required=True, help="Text file, one class name per line, in class_id order")
    parser.add_argument("--output", required=True, help="Output dataset folder (created/overwritten)")
    parser.add_argument("--train-split", type=float, default=None, help="Override train fraction (default from config.yaml)")
    parser.add_argument("--val-split", type=float, default=None, help="Override val fraction (default from config.yaml)")
    parser.add_argument("--test-split", type=float, default=None, help="Override test fraction (default from config.yaml)")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed (default from config.yaml)")
    parser.add_argument("--copy", action="store_true", default=True, help="Copy files (default). Mutually exclusive with --move.")
    parser.add_argument("--move", action="store_true", help="Move files instead of copying (saves disk space)")
    return parser.parse_args()


def load_class_names(classes_path: Path) -> list[str]:
    names = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not names:
        raise ValueError(f"No class names found in {classes_path}")
    return names


def find_matched_pairs(images_dir: Path, labels_dir: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    missing_labels: list[str] = []

    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    for img_path in image_paths:
        label_path = labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            pairs.append((img_path, label_path))
        else:
            missing_labels.append(img_path.name)

    if missing_labels:
        logger.warning(
            f"{len(missing_labels)} image(s) have no matching label file and will be SKIPPED: "
            f"{missing_labels[:10]}{' ...' if len(missing_labels) > 10 else ''}"
        )

    if not pairs:
        raise ValueError("No image/label pairs found. Check --images and --labels point to matching files.")

    return pairs


def split_dataset(
    pairs: list[tuple[Path, Path]], train_frac: float, val_frac: float, test_frac: float, seed: int,
) -> dict[str, list[tuple[Path, Path]]]:
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"train/val/test splits must sum to 1.0, got {total}")

    shuffled = pairs.copy()
    random.Random(seed).shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }


def write_split(split_name: str, items: list[tuple[Path, Path]], output_dir: Path, move: bool) -> None:
    images_out = output_dir / "images" / split_name
    labels_out = output_dir / "labels" / split_name
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    transfer = shutil.move if move else shutil.copy2

    for img_path, label_path in items:
        transfer(str(img_path), str(images_out / img_path.name))
        transfer(str(label_path), str(labels_out / label_path.name))

    logger.info(f"[{split_name}] {len(items)} image/label pairs written to {images_out} / {labels_out}")


def write_data_yaml(output_dir: Path, class_names: list[str]) -> Path:
    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_names),
        "names": class_names,
    }
    data_yaml_path = output_dir / "data.yaml"
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)
    logger.info(f"Wrote {data_yaml_path}")
    return data_yaml_path


def main() -> None:
    args = parse_args()
    cfg = load_config()

    images_dir = resolve_path(args.images)
    labels_dir = resolve_path(args.labels)
    classes_path = resolve_path(args.classes)
    output_dir = resolve_path(args.output)
    move = args.move

    train_frac, val_frac, test_frac = (
        cfg.training.train_val_test_split if hasattr(cfg, "training") else [0.8, 0.1, 0.1]
    )
    if args.train_split is not None:
        train_frac = args.train_split
    if args.val_split is not None:
        val_frac = args.val_split
    if args.test_split is not None:
        test_frac = args.test_split
    seed = args.seed if args.seed is not None else (cfg.training.seed if hasattr(cfg, "training") else 42)

    class_names = load_class_names(classes_path)
    logger.info(f"Loaded {len(class_names)} classes: {class_names}")

    pairs = find_matched_pairs(images_dir, labels_dir)
    logger.info(f"Found {len(pairs)} valid image/label pairs")

    splits = split_dataset(pairs, train_frac, val_frac, test_frac, seed)

    if output_dir.exists():
        logger.warning(f"Output directory {output_dir} already exists; files may be overwritten.")
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, items in splits.items():
        if items:
            write_split(split_name, items, output_dir, move)
        else:
            logger.warning(f"Split '{split_name}' is empty (0 images) - check your split ratios / dataset size.")

    write_data_yaml(output_dir, class_names)

    logger.info(
        f"Dataset ready: {len(splits['train'])} train / {len(splits['val'])} val / "
        f"{len(splits['test'])} test images in {output_dir}"
    )
    logger.info(f"Next step: python -m training.train_yolo --data {output_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
