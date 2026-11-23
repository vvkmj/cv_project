import os
import cv2
import numpy as np
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

COLOR_MAP = {
    "car": (255, 191, 0),        # Deep sky blue / amber
    "motorcycle": (0, 215, 255), # Yellow/gold
    "bus": (255, 105, 180),      # Hot pink
    "truck": (147, 112, 219),    # Medium purple
    "person": (50, 205, 50),     # Lime green
    "bicycle": (0, 255, 127),    # Spring green
}

VIOLATION_COLORS = {
    "no_helmet": (0, 0, 255),       # Red
    "multiple_riders": (0, 69, 255), # Red-orange
    "wrong_way": (0, 0, 230),       # Bright red
}

class EvidenceGenerator:
    """
    Generates and saves visual evidentiary records for detected violations:
    - Original unadulterated keyframe.
    - Annotated evidence frame with bounding boxes, direction vectors, and overlays.
    - Close-up cropped region of the offending vehicle/rider.
    - Full annotated video with persistent overlays.
    """

    def __init__(
        self,
        base_dir: str = "evidence",
        original_dir: str = "evidence/original",
        annotated_dir: str = "evidence/annotated",
        crop_dir: str = "evidence/crops",
        video_dir: str = "evidence/videos"
    ):
        self.base_dir = base_dir
        self.original_dir = original_dir
        self.annotated_dir = annotated_dir
        self.crop_dir = crop_dir
        self.video_dir = video_dir

        for d in [self.original_dir, self.annotated_dir, self.crop_dir, self.video_dir]:
            os.makedirs(d, exist_ok=True)

    def draw_annotations(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        active_violations: List[Dict[str, Any]],
        frame_idx: int = 0,
        timestamp: float = 0.0,
        camera_id: str = "CAM-01"
    ) -> np.ndarray:
        """Draws professional HUD bounding boxes, tracking labels, and violation banners."""
        annotated = frame.copy()
        h, w = frame.shape[:2]

        # Draw top HUD bar
        cv2.rectangle(annotated, (0, 0), (w, 40), (20, 24, 33), -1)
        hud_text = f"CAM: {camera_id} | TIME: {timestamp:.2f}s | FRAME: {frame_idx} | VEHICLES: {len(detections)}"
        cv2.putText(annotated, hud_text, (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 225, 230), 2)

        # Draw detections
        for det in detections:
            bbox = det["bounding_box"]
            x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
            cls_name = det.get("object_class", "vehicle")
            track_id = det.get("tracking_id")
            conf = det.get("confidence", 0.0)

            color = COLOR_MAP.get(cls_name, (200, 200, 200))
            
            # Corner-accented or standard bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = f"#{track_id} {cls_name} {conf:.2f}" if track_id else f"{cls_name} {conf:.2f}"
            t_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - 18)), (x1 + t_size[0] + 6, max(0, y1)), color, -1)
            cv2.putText(annotated, label, (x1 + 3, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

        # Overlay active violations
        banner_y = 65
        for viol in active_violations:
            v_type = viol.get("violation_type", "violation")
            t_id = viol.get("tracking_id", "N/A")
            v_color = VIOLATION_COLORS.get(v_type, (0, 0, 255))

            v_title = f"VIOLATION: {v_type.replace('_', ' ').upper()} (Track #{t_id})"
            cv2.rectangle(annotated, (15, banner_y - 20), (min(w - 15, 450), banner_y + 8), v_color, -1)
            cv2.putText(annotated, v_title, (22, banner_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            banner_y += 36

        return annotated

    def save_evidence(
        self,
        violation_id: str,
        original_frame: np.ndarray,
        annotated_frame: np.ndarray,
        crop_box: Optional[Dict[str, float]] = None
    ) -> Dict[str, str]:
        """
        Saves original frame, annotated frame, and vehicle crop to disk.
        Returns dictionary of relative filesystem paths.
        """
        orig_filename = f"{violation_id}_orig.jpg"
        annot_filename = f"{violation_id}_annot.jpg"
        crop_filename = f"{violation_id}_crop.jpg"

        orig_full = os.path.join(self.original_dir, orig_filename)
        annot_full = os.path.join(self.annotated_dir, annot_filename)
        crop_full = os.path.join(self.crop_dir, crop_filename)

        # Save original frame
        cv2.imwrite(orig_full, original_frame)

        # Save annotated keyframe
        cv2.imwrite(annot_full, annotated_frame)

        # Extract and save crop
        if crop_box:
            h, w = original_frame.shape[:2]
            cx1 = max(0, int(crop_box["x1"]))
            cy1 = max(0, int(crop_box["y1"]))
            cx2 = min(w, int(crop_box["x2"]))
            cy2 = min(h, int(crop_box["y2"]))
            crop = original_frame[cy1:cy2, cx1:cx2]
            if crop.size > 0:
                cv2.imwrite(crop_full, crop)
            else:
                cv2.imwrite(crop_full, annotated_frame)
        else:
            cv2.imwrite(crop_full, annotated_frame)

        return {
            "evidence_path": orig_full,
            "annotated_evidence_path": annot_full,
            "crop_path": crop_full
        }

