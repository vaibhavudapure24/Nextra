"""
Animal Tracking Module.
"""

from tracking.tracker import TrackedObject, BaseTracker, get_tracker
from tracking.bytetrack_tracker import ByteTrackTracker
from tracking.deepsort_tracker import DeepSORTTracker
from tracking.trajectory import TrajectoryManager
from tracking.speed_estimator import SpeedEstimator

__all__ = [
    "TrackedObject",
    "BaseTracker",
    "get_tracker",
    "ByteTrackTracker",
    "DeepSORTTracker",
    "TrajectoryManager",
    "SpeedEstimator",
]
