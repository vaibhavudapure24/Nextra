"""
PyTorch Bidirectional LSTM model for wildlife behavior sequence classification.
"""

import torch
import torch.nn as nn
from typing import Optional


class BehaviorLSTM(nn.Module):
    """
    Bidirectional LSTM network with temporal pooling and classification head.
    Input shape: (batch_size, sequence_length, feature_dim)
    Output shape: (batch_size, num_classes)
    """
    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 11,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, input_dim)
        out, (hn, cn) = self.lstm(x)
        # Global temporal average pooling across sequence dimension
        pooled = torch.mean(out, dim=1)
        logits = self.fc(pooled)
        return logits


def load_behavior_lstm(
    weights_path: Optional[str] = None,
    input_dim: int = 12,
    hidden_dim: int = 128,
    num_layers: int = 2,
    num_classes: int = 11,
    device: str = "cpu",
) -> Optional[BehaviorLSTM]:
    """
    Initializes and loads trained weights if file exists, else returns None.
    """
    import os
    if not weights_path or not os.path.exists(weights_path):
        return None

    model = BehaviorLSTM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
    )
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
