"""
Modular sequence-based behavior recognition pipeline.
Supports PyTorch LSTM and Transformer inference when trained weights are supplied,
and falls back to transparent kinematic heuristics when models are untrained.
"""

from typing import List, Tuple, Optional, Dict, Any
import os
import torch
import torch.nn.functional as F
import numpy as np

from utils.logger import get_logger
from utils.device import select_device
from behavior.features import extract_frame_features, FEATURE_DIM
from behavior.sequence_buffer import SequenceBuffer
from behavior.lstm_model import load_behavior_lstm
from behavior.transformer_model import load_behavior_transformer

logger = get_logger(__name__)

BEHAVIOR_CLASSES = [
    "eating",
    "sleeping",
    "walking",
    "running",
    "resting",
    "playing",
    "aggression",
    "stress",
    "limping",
    "injury",
    "social_interaction",
]


class BehaviorClassifier:
    """
    Classifies wildlife behavior from tracked animal trajectories and kinematics.
    """
    def __init__(
        self,
        model_type: str = "lstm",
        weights_path: Optional[str] = None,
        sequence_length: int = 30,
        device: str = "auto",
        min_confidence: float = 0.50,
    ):
        self.model_type = model_type.lower()
        self.sequence_length = sequence_length
        self.min_confidence = min_confidence
        self.device = select_device(device)

        self.sequence_buffer = SequenceBuffer(
            sequence_length=sequence_length, feature_dim=FEATURE_DIM
        )

        self.model = None
        self.is_trained = False
        self._init_model(weights_path)

    def _init_model(self, weights_path: Optional[str]) -> None:
        path = weights_path or f"models/behavior/{self.model_type}_behavior.pt"
        if os.path.exists(path):
            try:
                if self.model_type == "transformer":
                    self.model = load_behavior_transformer(
                        weights_path=path, device=str(self.device)
                    )
                else:
                    self.model = load_behavior_lstm(
                        weights_path=path, device=str(self.device)
                    )
                if self.model is not None:
                    self.is_trained = True
                    logger.info(f"Loaded trained {self.model_type.upper()} behavior model from {path}")
            except Exception as e:
                logger.warning(f"Failed to load behavior weights from {path}: {e}")

        if not self.is_trained:
            logger.info(
                f"No pre-trained weights found at '{path}'. Operating in heuristic baseline mode. "
                "Train using training/train_behavior.py to activate deep learning classification."
            )

    def update_and_classify(
        self,
        tracking_id: int,
        curr_pos: Tuple[float, float],
        prev_pos: Optional[Tuple[float, float]],
        prev_prev_pos: Optional[Tuple[float, float]],
        bbox: Tuple[float, float, float, float],
        speed_mps: float = 0.0,
        neighbor_positions: Optional[List[Tuple[float, float]]] = None,
    ) -> Dict[str, Any]:
        """
        Updates sequence buffer and performs behavior classification.
        """
        feat = extract_frame_features(
            curr_pos=curr_pos,
            prev_pos=prev_pos,
            prev_prev_pos=prev_prev_pos,
            bbox=bbox,
            neighbor_positions=neighbor_positions,
        )

        buffer_ready = self.sequence_buffer.add(tracking_id, feat)

        if self.is_trained and buffer_ready and self.model is not None:
            seq = self.sequence_buffer.get_sequence(tracking_id)
            if seq is not None:
                return self._infer_neural(seq)

        # Fallback to transparent kinematic heuristics
        return self._infer_heuristic(speed_mps, feat, neighbor_positions)

    def _infer_neural(self, seq_array: np.ndarray) -> Dict[str, Any]:
        tensor = torch.from_numpy(seq_array).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        class_idx = int(np.argmax(probs))
        confidence = float(probs[class_idx])
        behavior_label = BEHAVIOR_CLASSES[class_idx]

        return {
            "behavior": behavior_label,
            "confidence": round(confidence, 4),
            "is_ai_model": True,
            "model_type": self.model_type,
            "probabilities": {cls: round(float(p), 4) for cls, p in zip(BEHAVIOR_CLASSES, probs)},
        }

    def _infer_heuristic(
        self,
        speed_mps: float,
        feat: np.ndarray,
        neighbor_positions: Optional[List[Tuple[float, float]]],
    ) -> Dict[str, Any]:
        """
        Observable rule-based kinematic estimation.
        """
        aspect_ratio = feat[4]
        social_dist = feat[10]

        if speed_mps > 4.5:
            label = "running"
            conf = 0.85
        elif speed_mps > 0.8:
            label = "walking"
            conf = 0.80
        elif speed_mps < 0.15:
            # Low speed - differentiate resting, sleeping, eating
            if aspect_ratio > 1.6:
                label = "resting"
                conf = 0.75
            else:
                label = "eating"
                conf = 0.65
        else:
            label = "resting"
            conf = 0.60

        # Social interaction check if close neighbor
        if social_dist < 0.08 and neighbor_positions:
            label = "social_interaction"
            conf = 0.70

        return {
            "behavior": label,
            "confidence": round(conf, 2),
            "is_ai_model": False,
            "model_type": "heuristic_kinematic",
            "probabilities": {label: round(conf, 2)},
        }
