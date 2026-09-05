import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import numpy as np

from tracking.speed_estimator import SpeedEstimator
from tracking.trajectory import TrajectoryManager
from tracking.tracker import get_tracker, TrackedObject
from detection.postprocessing import DetectionResult


def test_speed_estimator():
    estimator = SpeedEstimator(meters_per_pixel=0.1, fps=30.0)
    # Moving 30 pixels in 1 second (30 frames) => 3.0 meters / 1.0 sec = 3.0 m/s
    speed = estimator.estimate_speed(
        pos_prev=(100.0, 100.0),
        pos_curr=(130.0, 100.0),
        time_delta_sec=1.0,
    )
    assert pytest.approx(speed, 0.05) == 3.0
    assert pytest.approx(SpeedEstimator.mps_to_kmh(speed), 0.1) == 10.8


def test_trajectory_manager():
    tm = TrajectoryManager(max_trail_points=10, meters_per_pixel=0.05)
    tm.update(track_id=1, centroid=(100.0, 100.0))
    dist = tm.update(track_id=1, centroid=(120.0, 100.0))
    # 20 pixels * 0.05 = 1.0 meter
    assert pytest.approx(dist, 0.01) == 1.0
    assert len(tm.get_trail(1)) == 2

    # Heatmap test
    heatmap = tm.generate_heatmap(width=300, height=300)
    assert heatmap.shape == (300, 300, 3)


def test_bytetrack_tracker():
    tracker = get_tracker("bytetrack")
    det1 = DetectionResult.create(class_id=0, species="lion", confidence=0.9, bbox=(50, 50, 150, 150))
    tracks = tracker.update([det1])
    assert isinstance(tracks, list)


def test_deepsort_tracker():
    tracker = get_tracker("deepsort")
    det1 = DetectionResult.create(class_id=0, species="rhino", confidence=0.85, bbox=(100, 100, 200, 200))
    tracks = tracker.update([det1])
    assert isinstance(tracks, list)
