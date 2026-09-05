"""
Temporal sliding window buffer for tracked wildlife behavior sequences.
"""

from typing import Dict, List, Optional
import numpy as np


class SequenceBuffer:
    """
    Maintains a FIFO sequence buffer of temporal feature vectors per tracking ID.
    """
    def __init__(self, sequence_length: int = 30, feature_dim: int = 12):
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        # Dict[tracking_id -> List[np.ndarray]]
        self.buffers: Dict[int, List[np.ndarray]] = {}

    def add(self, tracking_id: int, feature: np.ndarray) -> bool:
        """
        Adds a single frame's feature vector.
        Returns True if buffer has reached the required sequence length, False otherwise.
        """
        if tracking_id not in self.buffers:
            self.buffers[tracking_id] = []

        buf = self.buffers[tracking_id]
        buf.append(feature.astype(np.float32))

        if len(buf) > self.sequence_length:
            buf.pop(0)

        return len(buf) >= self.sequence_length

    def get_sequence(self, tracking_id: int) -> Optional[np.ndarray]:
        """
        Returns (sequence_length, feature_dim) array if buffer is full, else None.
        """
        buf = self.buffers.get(tracking_id)
        if buf and len(buf) >= self.sequence_length:
            return np.stack(buf, axis=0)
        return None

    def clear_track(self, tracking_id: int) -> None:
        """Cleans up expired track buffer."""
        self.buffers.pop(tracking_id, None)

    def reset(self) -> None:
        """Clears all track buffers."""
        self.buffers.clear()
