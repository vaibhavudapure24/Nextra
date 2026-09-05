import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import torch
import numpy as np

from anomaly.anomaly_features import extract_anomaly_features
from anomaly.isolation_forest import IsolationForestDetector
from anomaly.one_class_svm import OneClassSVMDetector
from anomaly.autoencoder import AnomalyAutoencoder
from anomaly.anomaly_detector import AnomalyDetector
from database.models import AlertSeverity


def test_anomaly_feature_extraction():
    feat = extract_anomaly_features(
        speed_history=[1.0, 1.2, 1.1],
        aspect_ratios=[1.2, 1.22],
        distance_to_fence_m=50.0,
        nearest_neighbor_distance_m=12.0,
    )
    assert len(feat) == 8
    assert isinstance(feat, np.ndarray)


def test_isolation_forest():
    iforest = IsolationForestDetector()
    X = np.random.randn(50, 8)
    iforest.fit(X)
    score = iforest.score(X[0])
    assert 0.0 <= score <= 1.0


def test_one_class_svm():
    svm = OneClassSVMDetector()
    X = np.random.randn(50, 8)
    svm.fit(X)
    score = svm.score(X[0])
    assert 0.0 <= score <= 1.0


def test_autoencoder():
    ae = AnomalyAutoencoder(input_dim=8, latent_dim=4)
    dummy = torch.randn(4, 8)
    rec = ae(dummy)
    assert rec.shape == (4, 8)
    score = ae.compute_anomaly_score(dummy[0])
    assert 0.0 <= score <= 1.0


def test_anomaly_detector_fence_breach():
    ad = AnomalyDetector()
    # Negative distance to fence triggers fence breach CRITICAL alert
    res = ad.evaluate(
        tracking_id=1,
        speed_history=[2.0],
        aspect_ratios=[1.2],
        distance_to_fence_m=-5.0,
    )
    assert res is not None
    assert res["anomaly_type"] == "FENCE_ESCAPE"
    assert res["severity"] == AlertSeverity.CRITICAL
