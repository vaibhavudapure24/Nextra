"""
Activity level and duration estimator based on observable animal motion.
"""

from typing import Dict, Any


class ActivityEstimator:
    """
    Estimates animal activity indices from accumulated behavior durations and distance.
    """
    def estimate(
        self,
        behavior_durations: Dict[str, float],
        total_distance_m: float,
        observation_seconds: float,
    ) -> Dict[str, float]:
        """
        Computes normalized activity score [0.0 - 1.0] and duration breakdowns in minutes.
        """
        obs_min = max(0.01, observation_seconds / 60.0)

        rest_min = behavior_durations.get("resting", 0.0) / 60.0
        sleep_min = behavior_durations.get("sleeping", 0.0) / 60.0
        feed_min = behavior_durations.get("eating", 0.0) / 60.0
        walk_min = behavior_durations.get("walking", 0.0) / 60.0
        run_min = behavior_durations.get("running", 0.0) / 60.0
        play_min = behavior_durations.get("playing", 0.0) / 60.0

        active_time_min = walk_min + (run_min * 1.5) + (play_min * 1.2) + (feed_min * 0.5)
        # Normalized activity index [0.0, 1.0]
        activity_level = min(1.0, active_time_min / obs_min)

        return {
            "activity_level": round(activity_level, 3),
            "rest_duration_min": round(rest_min, 1),
            "sleeping_duration_min": round(sleep_min, 1),
            "feeding_duration_min": round(feed_min, 1),
            "walking_distance_m": round(total_distance_m, 2),
            "daily_movement_m": round(total_distance_m, 2),
        }
