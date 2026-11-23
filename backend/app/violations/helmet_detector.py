import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

LABELS = {
    "helmet": "helmet", "with_helmet": "helmet", "hardhat": "helmet", "hard_hat": "helmet",
    "no_helmet": "no_helmet", "without_helmet": "no_helmet", "no_hat": "no_helmet",
}


def head_match(candidate, head):
    x1, y1, x2, y2 = candidate
    left, top, right, bottom = (head[key] for key in ("x1", "y1", "x2", "y2"))
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    if not (left < center_x < right and top < center_y < bottom and x2 > x1 and y2 > y1):
        return 0.0
    containment = (max(0, min(x2, right) - max(x1, left)) * max(0, min(y2, bottom) - max(y1, top)))
    containment /= (x2 - x1) * (y2 - y1)
    if containment < 0.55:
        return 0.0
    centrality = 1 - abs(2 * center_x - left - right) / (right - left)
    return 0.75 * containment + 0.25 * centrality


class HelmetDetector:
    def __init__(self, weights_path="ml/weights/helmet_yolov8.pt", confidence_threshold=0.50, device=None):
        self.weights_path = weights_path
        self.confidence_threshold = confidence_threshold
        self.device = device or "cpu"
        self.model = None
        self.is_available = False
        self.model_version = "Unavailable (No Weights)"
        self.status_message = "Dedicated helmet detection weights not found. Helmet analysis is disabled."
        if not Path(weights_path).is_file():
            return
        try:
            from ultralytics import YOLO

            model = YOLO(weights_path)
            labels = {self.normalize(name) for name in model.names.values()}
            if model.task != "detect" or not {"helmet", "no_helmet"} <= {LABELS.get(label) for label in labels}:
                raise ValueError("Helmet weights must detect both helmet and no-helmet boxes")
            self.model = model
            self.is_available = True
            self.model_version = f"YOLOv8-Helmet-{Path(weights_path).name}"
            self.status_message = "Helmet model loaded and operational."
        except Exception:
            self.model_version = "Error Loading Weights"
            self.status_message = "Helmet weights could not be loaded or lack the required detection classes."
            logger.exception("Helmet model initialization failed")

    @staticmethod
    def normalize(label):
        return label.lower().strip().replace(" ", "_").replace("-", "_")

    def analyze_head_region(self, full_image, head_box):
        return self.analyze_heads(full_image, [head_box], [head_box])[0]

    def analyze_heads(self, image, heads, visible_heads):
        unknown = self.result("unknown", 0.0, self.status_message if not self.is_available else "Head status unclear.")
        output = [unknown.copy() for _ in heads]
        if not self.is_available or not heads:
            return output
        height, width = image.shape[:2]
        samples = []
        for index, head in enumerate(heads):
            left, top, right, bottom = (int(head[key]) for key in ("x1", "y1", "x2", "y2"))
            left, top, right, bottom = max(0, left), max(0, top), min(width, right), min(height, bottom)
            if right - left < 10 or bottom - top < 10:
                continue
            pad_x, pad_y = int((right - left) * 0.15), int((bottom - top) * 0.20)
            x_origin, y_origin = max(0, left - pad_x), max(0, top - pad_y)
            crop = image[y_origin:min(height, bottom + pad_y), x_origin:min(width, right + pad_x)]
            samples.append((index, x_origin, y_origin, crop))

        for offset in range(0, len(samples), 8):
            batch = samples[offset:offset + 8]
            options = dict(source=[sample[3] for sample in batch], imgsz=224, conf=0.20, iou=0.50,
                           device=self.device, verbose=False, max_det=12)
            try:
                results = self.model.predict(**options)
            except (RuntimeError, ValueError):
                if self.device == "cpu":
                    raise
                logger.warning("Helmet accelerator inference failed; retrying on CPU")
                self.device = "cpu"
                self.model.to("cpu")
                results = self.model.predict(**{**options, "device": "cpu"})
            for (index, left, top, _), result in zip(batch, results, strict=True):
                scores = {}
                head = heads[index]
                for row in result.boxes.data.cpu().tolist():
                    x1, y1, x2, y2, confidence, class_id = row
                    status = LABELS.get(self.normalize(result.names[int(class_id)]))
                    if status is None:
                        continue
                    candidate = (x1 + left, y1 + top, x2 + left, y2 + top)
                    ownership = head_match(candidate, head)
                    if not ownership or any(
                        other is not head and head_match(candidate, other) >= ownership - 0.10
                        for other in visible_heads
                    ):
                        continue
                    scores[status] = max(confidence, scores.get(status, 0.0))
                if scores:
                    status, confidence = max(scores.items(), key=lambda entry: entry[1])
                    runner_up = max((score for label, score in scores.items() if label != status), default=0.0)
                    if confidence >= self.confidence_threshold and confidence - runner_up >= 0.15:
                        output[index] = self.result(status, confidence, "Spatially matched head detection.")
        return output

    def result(self, status, confidence, explanation):
        return dict(status=status, confidence=confidence, explanation=explanation, is_available=self.is_available)

    def get_status(self):
        return dict(name="Dedicated Helmet Detector", weights_path=os.path.basename(self.weights_path),
                    version=self.model_version, is_available=self.is_available, device=self.device,
                    status_message=self.status_message, confidence_threshold=self.confidence_threshold)
