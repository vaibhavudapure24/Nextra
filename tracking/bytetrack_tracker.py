"""
ByteTrack tracking implementation for wildlife monitoring.
Uses two-stage association (high-confidence and low-confidence detections)
to maintain track persistence through partial occlusions and rapid animal movements.
"""

from typing import List, Optional, Dict
import datetime as dt
import numpy as np

from utils.logger import get_logger
from detection.postprocessing import DetectionResult
from tracking.tracker import BaseTracker, TrackedObject
from tracking.speed_estimator import SpeedEstimator
from tracking.trajectory import TrajectoryManager

logger = get_logger(__name__)


class ByteTrackTracker(BaseTracker):
    """
    ByteTrack implementation leveraging supervision ByteTrack with trajectory and speed integration.
    """
    def __init__(
        self,
        track_thresh: float = 0.45,
        match_thresh: float = 0.80,
        track_buffer: int = 60,
        frame_rate: int = 30,
        max_trail_length: int = 90,
    ):
        super().__init__(max_trail_length=max_trail_length)
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.frame_rate = frame_rate

        self.speed_estimator = SpeedEstimator(fps=float(frame_rate))
        self.trajectory_manager = TrajectoryManager(max_trail_points=max_trail_length)

        # Initialize underlying ByteTrack engine from supervision
        import supervision as sv
        self.sv_tracker = sv.ByteTrack(
            track_activation_threshold=self.track_thresh,
            lost_track_buffer=self.track_buffer,
            minimum_matching_threshold=self.match_thresh,
            frame_rate=self.frame_rate,
        )
        logger.info("Initialized ByteTrack tracker")

    def update(
        self,
        detections: List[DetectionResult],
        frame: Optional[np.ndarray] = None,
        frame_number: int = 0,
        timestamp: Optional[float] = None,
    ) -> List[TrackedObject]:
        ts = timestamp if timestamp is not None else dt.datetime.now(dt.timezone.utc).timestamp()

        if not detections:
            # Still update empty detection to allow lost track decrement
            import supervision as sv
            empty_sv = sv.Detections.empty()
            self.sv_tracker.update_with_detections(empty_sv)
            return []

        import supervision as sv

        xyxy = np.array([d.bbox for d in detections], dtype=np.float32)
        confidence = np.array([d.confidence for d in detections], dtype=np.float32)
        class_id = np.array([d.class_id for d in detections], dtype=int)

        sv_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
        )

        tracked_sv = self.sv_tracker.update_with_detections(sv_detections)

        active_results: List[TrackedObject] = []

        if tracked_sv.tracker_id is not None and len(tracked_sv.tracker_id) > 0:
            for i in range(len(tracked_sv)):
                trk_id = int(tracked_sv.tracker_id[i])
                box = tuple(map(float, tracked_sv.xyxy[i].tolist()))
                conf = float(tracked_sv.confidence[i]) if tracked_sv.confidence is not None else 0.8
                cls_id = int(tracked_sv.class_id[i]) if tracked_sv.class_id is not None else 0

                # Match species from original detections
                species_name = f"animal_{cls_id}"
                for d in detections:
                    if d.class_id == cls_id:
                        species_name = d.species
                        break

                cx = (box[0] + box[2]) / 2.0
                cy = (box[1] + box[3]) / 2.0
                curr_pos = (cx, cy)

                # Prior state
                prev_trk = self.active_tracks.get(trk_id)
                prev_pos = prev_trk.current_position if prev_trk else curr_pos
                first_seen = prev_trk.first_seen if prev_trk else ts
                prev_speed = prev_trk.speed if prev_trk else 0.0

                # Calculate speed and update trajectory
                speed = self.speed_estimator.estimate_speed(
                    pos_prev=prev_pos,
                    pos_curr=curr_pos,
                    previous_speed=prev_speed,
                )
                tot_dist = self.trajectory_manager.update(trk_id, curr_pos)
                trail = self.trajectory_manager.get_trail(trk_id)

                trk_obj = TrackedObject(
                    tracking_id=trk_id,
                    species=species_name,
                    first_seen=first_seen,
                    last_seen=ts,
                    current_position=curr_pos,
                    previous_position=prev_pos,
                    bbox=box,
                    confidence=conf,
                    speed=speed,
                    distance_travelled=tot_dist,
                    trajectory=trail,
                    class_id=cls_id,
                )
                self.active_tracks[trk_id] = trk_obj
                active_results.append(trk_obj)

        return active_results
