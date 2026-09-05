"""
PyTorch Transformer model for wildlife behavior sequence classification.
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class BehaviorTransformer(nn.Module):
    """
    Temporal Transformer Classifier for animal behavioral sequences.
    """
    def __init__(
        self,
        input_dim: int = 12,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        num_classes: int = 11,
        dim_feedforward: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        projected = self.input_proj(x)
        encoded = self.pos_encoder(projected)
        out = self.transformer_encoder(encoded)
        # Average pooling across sequence
        pooled = torch.mean(out, dim=1)
        logits = self.classifier(pooled)
        return logits


def load_behavior_transformer(
    weights_path: Optional[str] = None,
    input_dim: int = 12,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 3,
    num_classes: int = 11,
    device: str = "cpu",
) -> Optional[BehaviorTransformer]:
    import os
    if not weights_path or not os.path.exists(weights_path):
        return None

    model = BehaviorTransformer(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        num_classes=num_classes,
    )
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
