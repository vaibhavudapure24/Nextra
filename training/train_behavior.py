"""
PyTorch Training Script for Wildlife Behavior Models (LSTM and Transformer).
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from utils.logger import get_logger
from utils.device import select_device
from behavior.lstm_model import BehaviorLSTM
from behavior.transformer_model import BehaviorTransformer
from behavior.behavior_classifier import BEHAVIOR_CLASSES
from training.metrics import compute_classification_metrics

logger = get_logger(__name__)


def generate_synthetic_behavior_data(num_samples: int = 1000, seq_len: int = 30, feat_dim: int = 12):
    """
    Generates synthetic training sequences for demonstration and initialization.
    """
    X = np.random.randn(num_samples, seq_len, feat_dim).astype(np.float32)
    # Distinct patterns for running (high feature 5) vs sleeping (low feature 5)
    y = np.random.randint(0, len(BEHAVIOR_CLASSES), size=(num_samples,))
    for i in range(num_samples):
        cls_idx = y[i]
        if cls_idx == 3:  # running
            X[i, :, 5] += 3.0
        elif cls_idx == 1:  # sleeping
            X[i, :, 5] = 0.01

    return torch.from_numpy(X), torch.from_numpy(y).long()


def train_behavior_model(
    model_type: str = "lstm",
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 1e-3,
    output_path: str = "models/behavior/lstm_behavior.pt",
    device: str = "auto",
):
    dev = select_device(device)
    logger.info(f"Training {model_type.upper()} behavior model on {dev}...")

    # Data preparation
    X, y = generate_synthetic_behavior_data()
    n_train = int(len(X) * 0.8)
    train_ds = TensorDataset(X[:n_train], y[:n_train])
    val_ds = TensorDataset(X[n_train:], y[n_train:])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Initialize model
    if model_type.lower() == "transformer":
        model = BehaviorTransformer(input_dim=12, num_classes=len(BEHAVIOR_CLASSES))
    else:
        model = BehaviorLSTM(input_dim=12, num_classes=len(BEHAVIOR_CLASSES))

    model.to(dev)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(dev), batch_y.to(dev)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_y)

        train_loss /= len(train_ds)

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(dev), batch_y.to(dev)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                val_loss += loss.item() * len(batch_y)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(batch_y.cpu().numpy())

        val_loss /= len(val_ds)
        metrics = compute_classification_metrics(all_targets, all_preds)

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            logger.info(
                f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} "
                f"| Val F1: {metrics['f1_score']:.4f} | Precision: {metrics['precision']:.4f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torch.save(model.state_dict(), output_path)

    logger.info(f"Trained {model_type.upper()} model weights saved successfully to: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Wildlife Behavior Classifier")
    parser.add_argument("--model", type=str, default="lstm", choices=["lstm", "transformer"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    out_file = args.output or f"models/behavior/{args.model}_behavior.pt"
    train_behavior_model(model_type=args.model, epochs=args.epochs, output_path=out_file)
