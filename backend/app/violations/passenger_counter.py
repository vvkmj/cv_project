from typing import List, Dict, Any, Optional
from collections import defaultdict
import statistics
import logging

logger = logging.getLogger(__name__)

class PassengerCounter:
    """
    Evaluates occupant counts on motorcycles across consecutive frames.
    Uses temporal stabilization (median filter across buffer) to prevent false alerts.
    """

    def __init__(self, buffer_size: int = 5, min_consecutive_frames: int = 3):
        self.buffer_size = buffer_size
        self.min_consecutive_frames = min_consecutive_frames
        # Maps motorcycle track_id -> list of observed rider counts
        self.count_history: Dict[int, List[int]] = defaultdict(list)
        # Tracks already reported violations to deduplicate per vehicle track
        self.reported_tracks = set()

    def update(
        self,
        motorcycle_track_id: Optional[int],
        associated_riders: List[Dict[str, Any]],
        passenger_limit: int = 2
    ) -> Optional[Dict[str, Any]]:
        """
        Updates count history for the motorcycle track and checks for multiple-rider violation.
        Returns violation detail dict if threshold is consistently exceeded, else None.
        """
        if motorcycle_track_id is None:
            # Cannot track across frames without tracking ID
            return None

        current_count = len(associated_riders)
        self.count_history[motorcycle_track_id].append(current_count)

        # Retain only the most recent buffer_size observations
        if len(self.count_history[motorcycle_track_id]) > self.buffer_size:
            self.count_history[motorcycle_track_id].pop(0)

        history = self.count_history[motorcycle_track_id]

        # Require minimum number of observations before raising violation
        if len(history) < self.min_consecutive_frames:
            return None

        # Check deduplication: only 1 alert per vehicle track
        if motorcycle_track_id in self.reported_tracks:
            return None

        # Use median count over recent window to eliminate transient false detections
        stable_count = int(statistics.median(history))

        # Check if recent consecutive frames consistently violate the limit
        recent_violations = [c > passenger_limit for c in history[-self.min_consecutive_frames:]]
        if all(recent_violations) and stable_count > passenger_limit:
            self.reported_tracks.add(motorcycle_track_id)
            
            explanation = (
                f"Motorcycle (Track #{motorcycle_track_id}) carrying {stable_count} occupants. "
                f"Permitted passenger limit is {passenger_limit} (temporal verification passed over {len(history)} frames)."
            )
            
            return {
                "violation_type": "multiple_riders",
                "tracking_id": motorcycle_track_id,
                "occupant_count": stable_count,
                "permitted_limit": passenger_limit,
                "confidence": 0.90,
                "explanation": explanation,
                "associated_riders": associated_riders
            }

        return None

    def reset(self):
        self.count_history.clear()
        self.reported_tracks.clear()

