"""
Anomaly Model Training Pipeline (Isolation Forest, One-Class SVM, and Autoencoder).
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from utils.logger import get_logger
from utils.device import select_device
from anomaly.isolation_forest import IsolationForestDetector
from anomaly.one_class_svm import OneClassSVMDetector
from anomaly.autoencoder import AnomalyAutoencoder

logger = get_logger(__name__)


def generate_synthetic_anomaly_training_data(num_samples: int = 1500, input_dim: int = 8) -> np.ndarray:
    """
    Generates nominal baseline behavior samples for training one-class/unsupervised anomaly detectors.
    Features: [mean_speed, speed_var, speed_range, ar_change, inact_min, run_min, fence_dist, social_dist]
    """
    np.random.seed(42)
    # Nominal speed around 1.0 m/s with moderate variance
    speed = np.random.normal(1.2, 0.4, (num_samples, 1))
    speed_var = np.random.exponential(0.2, (num_samples, 1))
    speed_range = speed_var * 2.0
    ar_change = np.random.exponential(0.1, (num_samples, 1))
    inact = np.random.exponential(0.5, (num_samples, 1))
    run = np.random.exponential(0.1, (num_samples, 1))
    fence = np.random.uniform(0.3, 1.0, (num_samples, 1))
    social = np.random.uniform(5.0, 30.0, (num_samples, 1))

    X = np.hstack([speed, speed_var, speed_range, ar_change, inact, run, fence, social]).astype(np.float32)
    return X


def train_anomaly_models(
    output_dir: str = "models/anomaly",
    epochs_ae: int = 30,
    batch_size: int = 32,
    device: str = "auto",
):
    os.makedirs(output_dir, exist_ok=True)
    X = generate_synthetic_anomaly_training_data()
    logger.info(f"Generated {len(X)} baseline feature samples for anomaly detector training.")

    # 1. Isolation Forest
    logger.info("Training Isolation Forest anomaly model...")
    iforest = IsolationForestDetector(contamination=0.05, n_estimators=120)
    iforest.fit(X)
    iforest_path = os.path.join(output_dir, "isolation_forest.joblib")
    iforest.save(iforest_path)
    logger.info(f"Saved Isolation Forest to {iforest_path}")

    # 2. One-Class SVM
    logger.info("Training One-Class SVM anomaly model...")
    svm = OneClassSVMDetector(nu=0.05)
    svm.fit(X)
    svm_path = os.path.join(output_dir, "one_class_svm.joblib")
    svm.save(svm_path)
    logger.info(f"Saved One-Class SVM to {svm_path}")

    # 3. PyTorch Autoencoder
    dev = select_device(device)
    logger.info(f"Training Autoencoder on {dev}...")
    ae = AnomalyAutoencoder(input_dim=8, latent_dim=4).to(dev)
    tensor_x = torch.from_numpy(X).float()
    loader = DataLoader(TensorDataset(tensor_x, tensor_x), batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(ae.parameters(), lr=1e-3, weight_decay=1e-5)

    for epoch in range(epochs_ae):
        ae.train()
        total_loss = 0.0
        for bx, _ in loader:
            bx = bx.to(dev)
            optimizer.zero_grad()
            rec = ae(bx)
            loss = criterion(rec, bx)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(bx)
        total_loss /= len(X)
        if (epoch + 1) % 10 == 0:
            logger.info(f"AE Epoch [{epoch+1}/{epochs_ae}] - Reconstruction Loss: {total_loss:.6f}")

    ae_path = os.path.join(output_dir, "autoencoder.pt")
    torch.save(ae.state_dict(), ae_path)
    logger.info(f"Saved Autoencoder weights to {ae_path}")

    return iforest_path, svm_path, ae_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Anomaly Detection Models")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--output", type=str, default="models/anomaly")
    args = parser.parse_args()

    train_anomaly_models(output_dir=args.output, epochs_ae=args.epochs)
