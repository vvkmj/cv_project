import math
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class WrongWayDetector:
    """
    Direction Analysis Module for detecting wrong-way vehicular motion.
    Uses camera-specific legal direction vectors, trajectory history,
    minimum movement thresholds, and temporal confirmation.
    """

    def __init__(
        self,
        min_movement_pixels: float = 25.0,
        consecutive_frames_required: int = 4,
        cosine_threshold: float = -0.35,
        window_size: int = 8
    ):
        self.min_movement_pixels = min_movement_pixels
        self.consecutive_frames_required = consecutive_frames_required
        self.cosine_threshold = cosine_threshold
        self.window_size = window_size

        # Maps track_id -> consecutive wrong-way frame count
        self.consecutive_wrong_way_counts: Dict[int, int] = defaultdict(int)
        # Deduplication set of reported track IDs
        self.reported_tracks = set()

    def parse_expected_direction(
        self, expected_dir: Any
    ) -> Tuple[float, float, float]:
        """
        Parses camera direction configuration into normalized (dx, dy) and angle_deg.
        Supports dict: {"angle_deg": 90.0} or {"dx": 0.0, "dy": 1.0} or float angle.
        Default: (0.0, 1.0) Southbound (downwards).
        """
        if isinstance(expected_dir, dict):
            if "angle_deg" in expected_dir and expected_dir["angle_deg"] is not None:
                deg = float(expected_dir["angle_deg"])
                rad = math.radians(deg)
                return math.cos(rad), math.sin(rad), deg
            elif "dx" in expected_dir and "dy" in expected_dir:
                dx = float(expected_dir["dx"])
                dy = float(expected_dir["dy"])
                mag = math.hypot(dx, dy)
                if mag > 1e-5:
                    dx, dy = dx / mag, dy / mag
                deg = (math.degrees(math.atan2(dy, dx)) + 360) % 360
                return dx, dy, deg
        elif isinstance(expected_dir, (int, float)):
            deg = float(expected_dir)
            rad = math.radians(deg)
            return math.cos(rad), math.sin(rad), deg

        # Default: downwards flow (dx=0, dy=1, angle=90 deg)
        return 0.0, 1.0, 90.0

    def evaluate_track(
        self,
        track,
        expected_direction_config: Any
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluates a single Track's trajectory against the camera's expected direction.
        Returns violation dict if wrong-way driving is confirmed, else None.
        """
        if track.track_id in self.reported_tracks:
            return None

        # Exclude static objects or tracks with insufficient history
        mv = track.get_movement_vector(window=self.window_size)
        if mv is None:
            return None

        dx, dy, dist = mv

        # Filter out stationary / parked vehicles or micro camera shake
        if dist < self.min_movement_pixels:
            # Vehicle has barely moved, reset consecutive count
            self.consecutive_wrong_way_counts[track.track_id] = 0
            return None

        # Actual movement unit vector
        actual_dx = dx / dist
        actual_dy = dy / dist
        actual_angle = (math.degrees(math.atan2(dy, dx)) + 360) % 360

        # Camera expected direction unit vector
        exp_dx, exp_dy, exp_angle = self.parse_expected_direction(expected_direction_config)

        # Dot product (Cosine Similarity)
        cos_sim = (actual_dx * exp_dx) + (actual_dy * exp_dy)

        # Inverted / opposing direction: cos_sim <= cosine_threshold (e.g. <= -0.35 => angle diff > 110 deg)
        if cos_sim <= self.cosine_threshold:
            self.consecutive_wrong_way_counts[track.track_id] += 1
        else:
            # Decrement or reset to avoid accumulating sporadic noise
            self.consecutive_wrong_way_counts[track.track_id] = max(
                0, self.consecutive_wrong_way_counts[track.track_id] - 1
            )

        # Check consecutive observation confirmation
        consecutive_count = self.consecutive_wrong_way_counts[track.track_id]
        if consecutive_count >= self.consecutive_frames_required:
            self.reported_tracks.add(track.track_id)

            angle_diff = abs(actual_angle - exp_angle)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff

            # Violation confidence scales with how directly opposing the movement is
            violation_confidence = round(min(0.98, max(0.70, abs(cos_sim))), 3)

            explanation = (
                f"Vehicle (Track #{track.track_id}, class: {track.object_class}) driving against legal traffic direction. "
                f"Observed movement heading {actual_angle:.1f}° vs legal heading {exp_angle:.1f}° "
                f"(cosine similarity: {cos_sim:.2f}, angular deviation: {angle_diff:.1f}°). "
                f"Confirmed across {consecutive_count} consecutive frames with {dist:.1f}px displacement."
            )

            return {
                "violation_type": "wrong_way",
                "tracking_id": track.track_id,
                "object_class": track.object_class,
                "confidence": violation_confidence,
                "movement_vector": {
                    "dx": round(dx, 2),
                    "dy": round(dy, 2),
                    "distance": round(dist, 2),
                    "actual_angle": round(actual_angle, 1),
                    "expected_angle": round(exp_angle, 1),
                    "cosine_similarity": round(cos_sim, 3)
                },
                "explanation": explanation
            }

        return None

    def reset(self):
        self.consecutive_wrong_way_counts.clear()
        self.reported_tracks.clear()

