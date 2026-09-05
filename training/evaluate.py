"""
Evaluates a trained YOLO checkpoint on the validation or test split and
reports precision, recall, mAP50, mAP50-95, per-class metrics, and a
confusion matrix.

Usage:
    python -m training.evaluate --weights outputs/training_runs/wildlife_yolo/weights/best.pt \\
        --data datasets/wildlife_dataset/data.yaml

    python -m training.evaluate --weights best.pt --data data.yaml --split test

Outputs:
    - Console summary (precision/recall/mAP, overall + per-class)
    - outputs/evaluation/<run_name>/metrics.json  (machine-readable summary)
    - outputs/evaluation/<run_name>/confusion_matrix.png (from Ultralytics)
    - outputs/evaluation/<run_name>/PR_curve.png, F1_curve.png, etc.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.logger import get_logger
from utils.config_loader import resolve_path

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained YOLO wildlife detection model")
    parser.add_argument("--weights", required=True, help="Path to trained .pt weights (e.g. best.pt)")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--split", default="val", choices=["val", "test"], help="Which split to evaluate on")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold used during evaluation")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for NMS during evaluation")
    parser.add_argument("--device", type=str, default="cpu", help="cuda | cpu | 0 | mps")
    parser.add_argument("--output", type=str, default="outputs/evaluation", help="Directory to save the metrics report")
    parser.add_argument("--name", type=str, default="run", help="Subfolder name for this evaluation's outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from ultralytics import YOLO  # lazy import

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    output_dir = resolve_path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading model: {weights_path}")
    model = YOLO(str(weights_path))

    logger.info(f"Running evaluation on '{args.split}' split of {args.data} ...")
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        project=str(output_dir),
        name=args.name,
        exist_ok=True,
        plots=True,   # tells Ultralytics to save confusion_matrix.png, PR_curve.png, F1_curve.png, etc.
    )

    class_names = metrics.names  # {class_id: name}
    precision_per_class = metrics.box.p     # array, one value per class
    recall_per_class = metrics.box.r
    map50_per_class = metrics.box.ap50
    map_per_class = metrics.box.ap

    summary = {
        "weights": str(weights_path),
        "data": args.data,
        "split": args.split,
        "overall": {
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "mAP50": float(metrics.box.map50),
            "mAP50-95": float(metrics.box.map),
        },
        "per_class": [
            {
                "class_id": int(class_id),
                "name": name,
                "precision": float(precision_per_class[i]) if i < len(precision_per_class) else None,
                "recall": float(recall_per_class[i]) if i < len(recall_per_class) else None,
                "mAP50": float(map50_per_class[i]) if i < len(map50_per_class) else None,
                "mAP50-95": float(map_per_class[i]) if i < len(map_per_class) else None,
            }
            for i, (class_id, name) in enumerate(class_names.items())
        ],
        "artifacts_dir": str(Path(metrics.save_dir)),
    }

    metrics_json_path = Path(metrics.save_dir) / "metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ------------------------------------------------------------------
    # Console report
    # ------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info(f"EVALUATION RESULTS ({args.split} split)")
    logger.info("=" * 70)
    logger.info(f"Overall Precision : {summary['overall']['precision']:.4f}")
    logger.info(f"Overall Recall    : {summary['overall']['recall']:.4f}")
    logger.info(f"mAP@0.5           : {summary['overall']['mAP50']:.4f}")
    logger.info(f"mAP@0.5:0.95      : {summary['overall']['mAP50-95']:.4f}")
    logger.info("-" * 70)
    logger.info(f"{'Class':<20}{'Precision':<12}{'Recall':<12}{'mAP50':<12}{'mAP50-95':<12}")
    for row in summary["per_class"]:
        logger.info(
            f"{row['name']:<20}"
            f"{(row['precision'] or 0):<12.4f}"
            f"{(row['recall'] or 0):<12.4f}"
            f"{(row['mAP50'] or 0):<12.4f}"
            f"{(row['mAP50-95'] or 0):<12.4f}"
        )
    logger.info("-" * 70)
    logger.info(f"Confusion matrix, PR curves, and full report saved to: {metrics.save_dir}")
    logger.info(f"Machine-readable summary: {metrics_json_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
