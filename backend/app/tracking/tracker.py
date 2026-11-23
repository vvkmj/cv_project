import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def calculate_iou(boxA: List[float], boxB: List[float]) -> float:
    """Computes Intersection over Union (IoU) between two [x1, y1, x2, y2] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    boxBArea = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])

    union = boxAArea + boxBArea - interArea
    if union <= 0:
        return 0.0
    return interArea / union


class Track:
    """Represents a single tracked entity across consecutive frames."""

    def __init__(self, track_id: int, detection: Dict[str, Any]):
        self.track_id = track_id
        self.object_class = detection["object_class"]
        self.confidence = detection["confidence"]
        bbox = detection["bounding_box"]
        self.bbox = [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]]
        
        cx = (self.bbox[0] + self.bbox[2]) / 2.0
        cy = (self.bbox[1] + self.bbox[3]) / 2.0
        
        self.history: List[Tuple[int, float, float, float, List[float]]] = [
            (detection["frame_number"], detection["timestamp"], cx, cy, list(self.bbox))
        ]
        self.age = 1
        self.hits = 1
        self.time_since_update = 0
        self.confirmed = False

    def update(self, detection: Dict[str, Any]):
        self.confidence = detection["confidence"]
        bbox = detection["bounding_box"]
        self.bbox = [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]]
        cx = (self.bbox[0] + self.bbox[2]) / 2.0
        cy = (self.bbox[1] + self.bbox[3]) / 2.0

        self.history.append(
            (detection["frame_number"], detection["timestamp"], cx, cy, list(self.bbox))
        )
        del self.history[:-32]
        self.hits += 1
        self.time_since_update = 0
        if self.hits >= 2:
            self.confirmed = True

    def mark_missed(self):
        self.time_since_update += 1
        self.age += 1

    def get_center(self) -> Tuple[float, float]:
        return (self.bbox[0] + self.bbox[2]) / 2.0, (self.bbox[1] + self.bbox[3]) / 2.0

    def predicted_box(self, frame_number: int) -> List[float]:
        if len(self.history) < 2:
            return self.bbox
        previous, current = self.history[-2:]
        scale = min(3, max(0, frame_number - current[0])) / max(1, current[0] - previous[0])
        dx, dy = (current[2] - previous[2]) * scale, (current[3] - previous[3]) * scale
        return [self.bbox[0] + dx, self.bbox[1] + dy, self.bbox[2] + dx, self.bbox[3] + dy]

    def get_movement_vector(self, window: int = 6) -> Optional[Tuple[float, float, float]]:
        """
        Calculates (dx, dy, distance) over the most recent `window` history points.
        Returns None if insufficient history is available.
        """
        if len(self.history) < 2:
            return None

        # Take points from history within the window
        sub_history = self.history[-window:]
        if len(sub_history) < 2:
            return None

        start_x, start_y = sub_history[0][2], sub_history[0][3]
        end_x, end_y = sub_history[-1][2], sub_history[-1][3]

        dx = end_x - start_x
        dy = end_y - start_y
        dist = np.hypot(dx, dy)
        return dx, dy, dist


class MultiObjectTracker:
    """
    High-performance Spatial & IoU Multi-Object Tracker.
    Maintains persistent IDs and vehicle trajectory history.
    """

    def __init__(self, iou_threshold: float = 0.25, max_age: int = 15, min_hits: int = 2):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.next_track_id = 1
        self.tracks: List[Track] = []

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Matches detections to existing tracks, creates new tracks, and updates history.
        Assigns `tracking_id` to each matched detection and returns the updated list.
        """
        matched_tracks = set()
        matched_dets = set()

        if len(self.tracks) > 0 and len(detections) > 0:
            # Build IoU cost matrix
            iou_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)
            for t_idx, track in enumerate(self.tracks):
                predicted = track.predicted_box(detections[0]["frame_number"])
                for d_idx, det in enumerate(detections):
                    # Only match if object classes are compatible (e.g., both vehicle or same class)
                    if track.object_class == det["object_class"]:
                        bbox = det["bounding_box"]
                        det_box = [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]]
                        iou_matrix[t_idx, d_idx] = max(
                            calculate_iou(predicted, det_box), 0.90 * calculate_iou(track.bbox, det_box)
                        )

            scores = np.where(iou_matrix >= self.iou_threshold, 1 + iou_matrix, 0)
            track_indices, detection_indices = linear_sum_assignment(scores, maximize=True)
            for t_idx, d_idx in zip(track_indices, detection_indices):
                if not scores[t_idx, d_idx]:
                    continue
                matched_tracks.add(t_idx)
                matched_dets.add(d_idx)
                
                # Update track with detection
                self.tracks[t_idx].update(detections[d_idx])
                self.tracks[t_idx].confirmed = self.tracks[t_idx].hits >= self.min_hits
                detections[d_idx]["tracking_id"] = self.tracks[t_idx].track_id

        # Create new tracks for unmatched detections
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_dets:
                new_track = Track(self.next_track_id, det)
                self.next_track_id += 1
                self.tracks.append(new_track)
                new_track.confirmed = self.min_hits <= 1
                matched_tracks.add(len(self.tracks) - 1)
                det["tracking_id"] = new_track.track_id

        # Mark missed tracks and prune stale ones
        surviving_tracks: List[Track] = []
        for t_idx, track in enumerate(self.tracks):
            if t_idx not in matched_tracks:
                track.mark_missed()
            if track.time_since_update <= self.max_age:
                surviving_tracks.append(track)
        self.tracks = surviving_tracks

        return detections

    def get_track_by_id(self, track_id: int) -> Optional[Track]:
        for t in self.tracks:
            if t.track_id == track_id:
                return t
        return None

    def reset(self):
        self.tracks = []
        self.next_track_id = 1

