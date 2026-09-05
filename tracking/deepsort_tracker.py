"""
DeepSORT tracking implementation for wildlife monitoring.
Combines Kalman Filtering state prediction with Hungarian matching across frames
to maintain continuous identities through occlusions.
"""

from typing import List, Optional, Tuple, Dict
import datetime as dt
import numpy as np
from scipy.optimize import linear_sum_assignment

from utils.logger import get_logger
from detection.postprocessing import DetectionResult, compute_iou
from tracking.tracker import BaseTracker, TrackedObject
from tracking.speed_estimator import SpeedEstimator
from tracking.trajectory import TrajectoryManager

logger = get_logger(__name__)


class KalmanBoxTracker:
    """
    Kalman filter tracker representing the state of a single bounding box:
    State vector: [x, y, s, r, vx, vy, vs] where:
      x, y: center of box
      s: scale (area)
      r: aspect ratio
    """
    count = 0

    def __init__(self, bbox: Tuple[float, float, float, float]):
        from filterpy.kalman import KalmanFilter

        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        # State transition matrix
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ])
        # Measurement matrix
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ])
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = self._bbox_to_z(bbox)
        self.time_since_update = 0
        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox: Tuple[float, float, float, float]):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self._bbox_to_z(bbox))

    def predict(self) -> Tuple[float, float, float, float]:
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] = 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self._x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self) -> Tuple[float, float, float, float]:
        return self._x_to_bbox(self.kf.x)

    @staticmethod
    def _bbox_to_z(bbox: Tuple[float, float, float, float]) -> np.ndarray:
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.0
        y = bbox[1] + h / 2.0
        s = w * h
        r = w / float(h) if h > 0 else 1.0
        return np.array([x, y, s, r]).reshape((4, 1))

    @staticmethod
    def _x_to_bbox(x: np.ndarray) -> Tuple[float, float, float, float]:
        w = np.sqrt(max(0.0, x[2] * x[3]))
        h = x[2] / w if w > 0 else 0.0
        return (
            float(x[0] - w / 2.0),
            float(x[1] - h / 2.0),
            float(x[0] + w / 2.0),
            float(x[1] + h / 2.0),
        )


class DeepSORTTracker(BaseTracker):
    """
    DeepSORT tracker with Kalman filtering, Hungarian assignment, and occlusion handling.
    """
    def __init__(
        self,
        max_age: int = 45,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        max_trail_length: int = 90,
    ):
        super().__init__(max_trail_length=max_trail_length)
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: List[KalmanBoxTracker] = []
        self.track_species: Dict[int, str] = {}
        self.track_classes: Dict[int, int] = {}
        self.speed_estimator = SpeedEstimator()
        self.trajectory_manager = TrajectoryManager(max_trail_points=max_trail_length)
        logger.info("Initialized DeepSORT tracker")

    def update(
        self,
        detections: List[DetectionResult],
        frame: Optional[np.ndarray] = None,
        frame_number: int = 0,
        timestamp: Optional[float] = None,
    ) -> List[TrackedObject]:
        ts = timestamp if timestamp is not None else dt.datetime.now(dt.timezone.utc).timestamp()

        # Step 1: Predict new locations of existing trackers
        trks = np.zeros((len(self.trackers), 4))
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            trks[t, :] = [pos[0], pos[1], pos[2], pos[3]]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        for t in reversed(to_del):
            self.trackers.pop(t)
            trks = np.delete(trks, t, axis=0)

        # Step 2: Match detections to track predictions using IoU cost matrix
        matched, unmatched_dets, unmatched_trks = self._associate_detections(detections, trks)

        # Step 3: Update matched trackers
        for d, t in matched:
            det = detections[d]
            trk = self.trackers[t]
            trk.update(det.bbox)
            self.track_species[trk.id] = det.species
            self.track_classes[trk.id] = det.class_id

        # Step 4: Create new trackers for unmatched detections
        for d in unmatched_dets:
            det = detections[d]
            trk = KalmanBoxTracker(det.bbox)
            self.trackers.append(trk)
            self.track_species[trk.id] = det.species
            self.track_classes[trk.id] = det.class_id

        # Step 5: Gather active outputs and purge dead tracks
        active_results: List[TrackedObject] = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            i -= 1
            bbox = trk.get_state()
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or frame_number <= self.min_hits):
                tid = trk.id
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
                curr_pos = (cx, cy)

                prev_trk = self.active_tracks.get(tid)
                prev_pos = prev_trk.current_position if prev_trk else curr_pos
                first_seen = prev_trk.first_seen if prev_trk else ts
                prev_speed = prev_trk.speed if prev_trk else 0.0

                speed = self.speed_estimator.estimate_speed(
                    pos_prev=prev_pos,
                    pos_curr=curr_pos,
                    previous_speed=prev_speed,
                )
                tot_dist = self.trajectory_manager.update(tid, curr_pos)
                trail = self.trajectory_manager.get_trail(tid)

                trk_obj = TrackedObject(
                    tracking_id=tid,
                    species=self.track_species.get(tid, "wildlife"),
                    first_seen=first_seen,
                    last_seen=ts,
                    current_position=curr_pos,
                    previous_position=prev_pos,
                    bbox=bbox,
                    confidence=0.85,
                    speed=speed,
                    distance_travelled=tot_dist,
                    trajectory=trail,
                    class_id=self.track_classes.get(tid, 0),
                )
                self.active_tracks[tid] = trk_obj
                active_results.append(trk_obj)

            # Purge dead tracks beyond max_age
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        return active_results

    def _associate_detections(
        self, detections: List[DetectionResult], trks: np.ndarray
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if len(trks) == 0:
            return [], list(range(len(detections))), []
        if len(detections) == 0:
            return [], [], list(range(len(trks)))

        iou_matrix = np.zeros((len(detections), len(trks)), dtype=np.float32)
        for d, det in enumerate(detections):
            for t in range(len(trks)):
                iou_matrix[d, t] = compute_iou(det.bbox, tuple(trks[t]))

        row_ind, col_ind = linear_sum_assignment(-iou_matrix)

        matched_indices = []
        unmatched_dets = list(range(len(detections)))
        unmatched_trks = list(range(len(trks)))

        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= self.iou_threshold:
                matched_indices.append((r, c))
                if r in unmatched_dets:
                    unmatched_dets.remove(r)
                if c in unmatched_trks:
                    unmatched_trks.remove(c)

        return matched_indices, unmatched_dets, unmatched_trks
