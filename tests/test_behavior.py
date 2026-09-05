import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import torch
import numpy as np

from behavior.features import extract_frame_features, FEATURE_DIM
from behavior.sequence_buffer import SequenceBuffer
from behavior.lstm_model import BehaviorLSTM
from behavior.transformer_model import BehaviorTransformer
from behavior.behavior_classifier import BehaviorClassifier


def test_feature_extraction():
    feat = extract_frame_features(
        curr_pos=(640.0, 360.0),
        prev_pos=(630.0, 360.0),
        prev_prev_pos=(620.0, 360.0),
        bbox=(600.0, 300.0, 680.0, 420.0),
    )
    assert len(feat) == FEATURE_DIM
    assert isinstance(feat, np.ndarray)


def test_sequence_buffer():
    buf = SequenceBuffer(sequence_length=5, feature_dim=12)
    sample_feat = np.ones(12, dtype=np.float32)

    for i in range(4):
        ready = buf.add(tracking_id=1, feature=sample_feat)
        assert not ready

    ready = buf.add(tracking_id=1, feature=sample_feat)
    assert ready

    seq = buf.get_sequence(tracking_id=1)
    assert seq.shape == (5, 12)


def test_lstm_forward_pass():
    model = BehaviorLSTM(input_dim=12, hidden_dim=32, num_layers=1, num_classes=11)
    dummy_input = torch.randn(2, 30, 12)
    output = model(dummy_input)
    assert output.shape == (2, 11)


def test_transformer_forward_pass():
    model = BehaviorTransformer(input_dim=12, d_model=32, nhead=2, num_layers=2, num_classes=11)
    dummy_input = torch.randn(2, 30, 12)
    output = model(dummy_input)
    assert output.shape == (2, 11)


def test_behavior_classifier():
    classifier = BehaviorClassifier(device="cpu")
    result = classifier.update_and_classify(
        tracking_id=1,
        curr_pos=(500, 500),
        prev_pos=(490, 500),
        prev_prev_pos=None,
        bbox=(450, 450, 550, 550),
        speed_mps=1.5,
    )
    assert "behavior" in result
    assert "confidence" in result
    assert "is_ai_model" in result
