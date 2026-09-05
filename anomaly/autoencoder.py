"""
PyTorch Autoencoder for anomaly detection via feature reconstruction loss.
"""

from typing import Optional
import os
import torch
import torch.nn as nn
import numpy as np


class AnomalyAutoencoder(nn.Module):
    """
    Symmetric autoencoder compressing features into a latent bottleneck.
    Reconstruction loss (MSE) serves as the raw anomaly score.
    """
    def __init__(self, input_dim: int = 8, latent_dim: int = 4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction

    def compute_anomaly_score(self, x: torch.Tensor) -> float:
        self.eval()
        with torch.no_grad():
            if x.dim() == 1:
                x = x.unsqueeze(0)
            rec = self.forward(x)
            mse = torch.mean((x - rec) ** 2).item()
            # Normalize to 0-1 range sigmoid proxy
            score = 1.0 / (1.0 + np.exp(-mse * 5.0 + 2.5))
            return float(score)


def load_autoencoder(weights_path: str, input_dim: int = 8, device: str = "cpu") -> Optional[AnomalyAutoencoder]:
    if not os.path.exists(weights_path):
        return None
    model = AnomalyAutoencoder(input_dim=input_dim)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
