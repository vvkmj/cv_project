import os
import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

class HelmetDetector:
    """
    Dedicated Helmet Detection & Classification Module.
    Evaluates extracted rider head regions.
    Supports classes: 'helmet', 'no_helmet', 'unknown'.
    Strictly reports unavailability if dedicated weights are not present.
    """

    def __init__(
        self,
        weights_path: str = "ml/weights/helmet_yolov8.pt",
        confidence_threshold: float = 0.50,
        device: Optional[str] = None
    ):
        self.weights_path = weights_path
        self.confidence_threshold = confidence_threshold
        self.device = device or "cpu"
        self.model = None
        self.is_available = False
        self.model_version = "None"
        self.status_message = ""
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.weights_path):
            self.is_available = False
            self.model = None
            self.model_version = "Unavailable (No Weights)"
            self.status_message = (
                f"Dedicated helmet detection weights '{self.weights_path}' not found. "
                "Helmet analysis is currently DISABLED. Place trained YOLO weights in ml/weights/ "
                "or run python ml/training/train_helmet.py to enable."
            )
            logger.info(self.status_message)
            return

        try:
            from ultralytics import YOLO
            self.model = YOLO(self.weights_path)
            self.is_available = True
            self.model_version = f"YOLOv8-Helmet-{os.path.basename(self.weights_path)}"
            self.status_message = "Helmet model loaded and operational."
            logger.info(f"Loaded dedicated helmet model from {self.weights_path}")
        except Exception as e:
            self.is_available = False
            self.model = None
            self.model_version = "Error Loading Weights"
            self.status_message = f"Failed to load helmet model from {self.weights_path}: {e}"
            logger.error(self.status_message)

    def analyze_head_region(
        self,
        full_image: np.ndarray,
        head_box: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Analyzes the rider head region.
        Returns:
            status: 'helmet' | 'no_helmet' | 'unknown'
            confidence: float (0.0 to 1.0)
            explanation: str
        """
        if not self.is_available or self.model is None:
            return {
                "status": "unknown",
                "confidence": 0.0,
                "explanation": self.status_message,
                "is_available": False
            }

        hx1 = max(0, int(head_box["x1"]))
        hy1 = max(0, int(head_box["y1"]))
        hx2 = min(full_image.shape[1], int(head_box["x2"]))
        hy2 = min(full_image.shape[0], int(head_box["y2"]))

        if (hx2 - hx1) < 10 or (hy2 - hy1) < 10:
            return {
                "status": "unknown",
                "confidence": 0.0,
                "explanation": "Head region too small or occluded for reliable helmet verification.",
                "is_available": True
            }

        # Slightly expand crop boundary to provide context for headgear/hair detection
        pad_x = int((hx2 - hx1) * 0.20)
        pad_y = int((hy2 - hy1) * 0.20)
        cx1 = max(0, hx1 - pad_x)
        cy1 = max(0, hy1 - pad_y)
        cx2 = min(full_image.shape[1], hx2 + pad_x)
        cy2 = min(full_image.shape[0], hy2 + pad_y)

        crop = full_image[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return {
                "status": "unknown",
                "confidence": 0.0,
                "explanation": "Empty crop for head region.",
                "is_available": True
            }

        try:
            results = self.model.predict(
                source=crop,
                imgsz=224,
                conf=0.20,
                device=self.device,
                verbose=False
            )

            if not results or len(results) == 0 or len(results[0].boxes) == 0:
                return {
                    "status": "unknown",
                    "confidence": 0.0,
                    "explanation": "No helmet or head features detected in crop with sufficient confidence.",
                    "is_available": True
                }

            boxes = results[0].boxes
            best_conf = 0.0
            best_status = "unknown"

            # Check detected classes
            # Expected custom classes: 0: 'helmet', 1: 'no_helmet' (or names from model)
            names = results[0].names
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                label = names.get(cls_id, "").lower()
                norm_label = label.replace(" ", "_").replace("-", "_")

                if any(k in norm_label for k in ["without_helmet", "no_helmet", "without", "no_hat"]):
                    if conf > best_conf:
                        best_conf = conf
                        best_status = "no_helmet"
                elif any(k in norm_label for k in ["with_helmet", "helmet", "hardhat", "hard_hat"]):
                    if conf > best_conf:
                        best_conf = conf
                        best_status = "helmet"

            if best_conf < self.confidence_threshold:
                return {
                    "status": "unknown",
                    "confidence": round(best_conf, 3),
                    "explanation": f"Helmet classification confidence ({best_conf:.2f}) below threshold ({self.confidence_threshold:.2f}). Marked as unknown.",
                    "is_available": True
                }

            explanation = (
                f"Rider head evaluated: {best_status.replace('_', ' ').title()} "
                f"with confidence {best_conf:.2f}."
            )
            return {
                "status": best_status,
                "confidence": round(best_conf, 3),
                "explanation": explanation,
                "is_available": True
            }

        except Exception as e:
            logger.error(f"Error in helmet inference on crop: {e}")
            return {
                "status": "unknown",
                "confidence": 0.0,
                "explanation": f"Inference error during helmet analysis: {e}",
                "is_available": True
            }

    def get_status(self) -> Dict[str, Any]:
        return {
            "name": "Dedicated Helmet Detector",
            "weights_path": self.weights_path,
            "version": self.model_version,
            "is_available": self.is_available,
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "status_message": self.status_message,
            "guidance": (
                "To enable automated no-helmet violations, provide custom YOLOv8 weights trained "
                "on a helmet dataset (e.g. Roboflow/Kaggle Helmet Dataset) saved to ml/weights/helmet_yolov8.pt."
            )
        }

