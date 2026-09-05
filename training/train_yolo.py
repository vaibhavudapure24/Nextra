"""
Fine-tunes YOLO11 (or YOLOv8) on a custom wildlife dataset prepared by
datasets/prepare_dataset.py.

Usage:
    python -m training.train_yolo --data datasets/wildlife_dataset/data.yaml
    python -m training.train_yolo --data datasets/wildlife_dataset/data.yaml \\
        --model yolov8s.pt --epochs 150 --batch 32 --device 0

All defaults come from configs/config.yaml -> training.*, and can be
overridden per-run via CLI flags.

After training finishes, this script:
  1. Prints the path to the best checkpoint (best.pt).
  2. Automatically updates configs/species_map.yaml -> custom_wildlife with
     the trained class names, so the rest of the system (detector, API,
     dashboard) immediately shows correct species names.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from utils.logger import get_logger
from utils.config_loader import load_config, resolve_path

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    cfg = load_config()
    t_cfg = cfg.training

    parser = argparse.ArgumentParser(description="Train/fine-tune YOLO11 or YOLOv8 on a wildlife dataset")
    parser.add_argument("--data", type=str, default=str(resolve_path(t_cfg.data_yaml)), help="Path to data.yaml")
    parser.add_argument("--model", type=str, default=t_cfg.base_model, help="Base checkpoint to fine-tune, e.g. yolo11n.pt, yolo11s.pt, yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=t_cfg.epochs)
    parser.add_argument("--imgsz", type=int, default=t_cfg.imgsz)
    parser.add_argument("--batch", type=int, default=t_cfg.batch)
    parser.add_argument("--patience", type=int, default=t_cfg.patience)
    parser.add_argument("--device", type=str, default=t_cfg.device, help="cuda | cpu | 0 | 0,1 | mps")
    parser.add_argument("--workers", type=int, default=t_cfg.workers)
    parser.add_argument("--project", type=str, default=str(resolve_path(t_cfg.project_dir)))
    parser.add_argument("--name", type=str, default=t_cfg.run_name)
    parser.add_argument("--resume", action="store_true", help="Resume from the last checkpoint of --name if it exists")
    parser.add_argument("--no-species-map-update", action="store_true", help="Skip auto-updating configs/species_map.yaml")
    return parser.parse_args()


def update_species_map(class_names: list[str], species_map_path: Path) -> None:
    """Populate configs/species_map.yaml -> custom_wildlife with the newly trained class names."""
    if not species_map_path.exists():
        logger.warning(f"{species_map_path} not found; skipping species map update.")
        return

    with open(species_map_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    raw["custom_wildlife"] = {i: name for i, name in enumerate(class_names)}

    with open(species_map_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(raw, f, sort_keys=False, allow_unicode=True)

    logger.info(f"Updated {species_map_path} -> custom_wildlife with {len(class_names)} trained classes.")


def main() -> None:
    args = parse_args()

    from ultralytics import YOLO  # lazy import: keeps `--help` fast and avoids torch import for non-training CLI use

    data_yaml_path = Path(args.data)
    if not data_yaml_path.exists():
        raise FileNotFoundError(
            f"data.yaml not found at {data_yaml_path}. Run datasets/prepare_dataset.py first — see datasets/README.md."
        )

    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    class_names = list(data_cfg["names"]) if isinstance(data_cfg["names"], (list, tuple)) else list(data_cfg["names"].values())

    logger.info(f"Starting training: model={args.model}, data={data_yaml_path}, classes={class_names}")
    logger.info(f"epochs={args.epochs} imgsz={args.imgsz} batch={args.batch} device={args.device}")

    model = YOLO(args.model)

    results = model.train(
        data=str(data_yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        resume=args.resume,
        exist_ok=True,
    )

    run_dir = Path(results.save_dir)
    best_weights = run_dir / "weights" / "best.pt"

    logger.info("=" * 70)
    logger.info("Training complete.")
    logger.info(f"Best weights: {best_weights}")
    logger.info(f"Training run artifacts (loss curves, PR curves, sample batches): {run_dir}")
    logger.info("=" * 70)
    logger.info("To use this model, set in configs/config.yaml:")
    logger.info(f'  detection.model_path: "{best_weights}"')
    logger.info(f'  detection.model_type: "{"yolo11" if "yolo11" in args.model else "yolov8"}"')

    if not args.no_species_map_update:
        species_map_path = resolve_path(load_config().detection.species_map_path)
        update_species_map(class_names, species_map_path)

    logger.info("Next step: evaluate the model -> python -m training.evaluate "
                f"--weights {best_weights} --data {data_yaml_path}")


if __name__ == "__main__":
    main()
