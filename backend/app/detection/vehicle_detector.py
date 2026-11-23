import os
import logging
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

# COCO 80 class IDs for traffic relevant entities
TARGET_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

class VehicleDetector:
    """
    YOLO-based Vehicle & Person Detector using Ultralytics YOLOv8.
    Detects motorcycles, cars, buses, trucks, people, and bicycles.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        device: Optional[str] = None
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        self.is_loaded = False
        self.device = device or self._detect_device()
        self.model_version = f"YOLOv8-{os.path.basename(model_path)}"
        self._load_model()

    def _detect_device(self) -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    def _load_model(self):
        try:
            from ultralytics import YOLO
            logger.info(f"Loading vehicle detection model: {self.model_path} on {self.device}")
            self.model = YOLO(self.model_path)
            self.is_loaded = True
            logger.info("Vehicle detection model loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not initialize YOLO model ({self.model_path}): {e}")
            self.model = None
            self.is_loaded = False

    def detect(self, image: np.ndarray, frame_number: int = 0, timestamp: float = 0.0) -> List[Dict[str, Any]]:
        """
        Runs object detection on a BGR image frame.
        Returns a list of raw detection dictionaries conforming to the Detection contract.
        """
        if not self.is_loaded or self.model is None:
            logger.debug("VehicleDetector: Model is not loaded.")
            return []

        try:
            # Run inference targeting only classes of interest
            results = self.model.predict(
                source=image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                device=self.device,
                classes=list(TARGET_CLASSES.keys()),
                verbose=False
            )

            detections: List[Dict[str, Any]] = []
            if not results or len(results) == 0:
                return detections

            result = results[0]
            boxes = result.boxes

            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].cpu().numpy().tolist()

                cls_name = TARGET_CLASSES.get(cls_id, result.names.get(cls_id, "unknown"))

                detections.append({
                    "frame_number": frame_number,
                    "timestamp": timestamp,
                    "tracking_id": None,
                    "object_class": cls_name,
                    "confidence": round(conf, 4),
                    "bounding_box": {
                        "x1": round(xyxy[0], 2),
                        "y1": round(xyxy[1], 2),
                        "x2": round(xyxy[2], 2),
                        "y2": round(xyxy[3], 2),
                    },
                    "model_version": self.model_version
                })

            return detections
        except Exception as e:
            logger.error(f"Error during detection on frame {frame_number}: {e}", exc_info=True)
            return []

    def get_status(self) -> Dict[str, Any]:
        return {
            "name": "Vehicle & Person Detector",
            "model_path": self.model_path,
            "version": self.model_version,
            "is_loaded": self.is_loaded,
            "device": self.device,
            "target_classes": list(TARGET_CLASSES.values()),
            "confidence_threshold": self.confidence_threshold
        }

