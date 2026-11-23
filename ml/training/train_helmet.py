import os
import sys
import shutil
import yaml
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainHelmet")

def train_helmet_model(
    config_path: str = "ml/configs/helmet_training.yaml",
    base_model: str = "yolov8n.pt",
    output_weights_path: str = "ml/weights/helmet_yolov8.pt",
    epochs: int = 30,
    batch_size: int = 16,
    imgsz: int = 640,
    device: Optional[str] = None
):
    """
    Executes reproducible transfer learning training for helmet violation detection.
    Saves best model weights to `ml/weights/helmet_yolov8.pt`.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Training configuration file '{config_path}' not found.")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Check dataset existence
    data_path = cfg.get("path", "")
    full_data_path = os.path.normpath(os.path.join(os.path.dirname(config_path), data_path))
    train_images = os.path.join(full_data_path, cfg.get("train", "images/train"))

    if not os.path.exists(train_images) or len(os.listdir(train_images)) == 0:
        logger.warning(
            f"Dataset train images not found at '{train_images}'.\n"
            "To train custom weights, download a verified dataset (e.g., Roboflow Helmet Dataset) "
            "and run `python ml/datasets/dataset_prep.py` before training."
        )
        return False

    try:
        from ultralytics import YOLO
        import torch

        # Device detection
        if device is None:
            if torch.cuda.is_available():
                device = "0"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        logger.info(f"Initializing YOLO transfer learning with base weights '{base_model}' on device '{device}'...")
        model = YOLO(base_model)

        # Train model
        results = model.train(
            data=config_path,
            epochs=epochs or cfg.get("epochs", 30),
            batch=batch_size or cfg.get("batch", 16),
            imgsz=imgsz or cfg.get("imgsz", 640),
            device=device,
            optimizer=cfg.get("optimizer", "AdamW"),
            lr0=cfg.get("lr0", 0.01),
            patience=cfg.get("patience", 10),
            project="ml/runs",
            name="helmet_detection_experiment",
            exist_ok=True,
            verbose=True
        )

        # Copy best weights to target destination
        best_weights = os.path.join("ml/runs/helmet_detection_experiment/weights/best.pt")
        if os.path.exists(best_weights):
            os.makedirs(os.path.dirname(output_weights_path), exist_ok=True)
            shutil.copy2(best_weights, output_weights_path)
            logger.info(f"Successfully exported best model weights to: {output_weights_path}")
            return True
        else:
            logger.warning("Training finished but best.pt was not found at standard path.")
            return False

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train custom YOLOv8 helmet detection model")
    parser.add_argument("--config", default="ml/configs/helmet_training.yaml", help="Path to data YAML")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", default=None, help="Device (cpu, mps, 0)")
    args = parser.parse_args()

    train_helmet_model(
        config_path=args.config,
        epochs=args.epochs,
        batch_size=args.batch,
        device=args.device
    )

