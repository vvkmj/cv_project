import os
import shutil
import random
import json
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Any
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DatasetPrep")

class DatasetPreparer:
    """
    Standardized dataset ingestion, conversion, validation, and split utility
    for YOLOv8 traffic and helmet datasets.
    """

    def __init__(self, target_dir: str = "ml/datasets/helmet_dataset"):
        self.target_dir = target_dir
        self.classes = ["helmet", "no_helmet"]
        self.class_to_id = {cls: idx for idx, cls in enumerate(self.classes)}

    def create_directory_structure(self):
        """Creates standard YOLOv8 folder layout."""
        for split in ["train", "val", "test"]:
            os.makedirs(os.path.join(self.target_dir, "images", split), exist_ok=True)
            os.makedirs(os.path.join(self.target_dir, "labels", split), exist_ok=True)
        logger.info(f"Initialized directory structure at: {self.target_dir}")

    def validate_yolo_label(self, label_path: str) -> Tuple[bool, int, str]:
        """
        Validates that a YOLO label file contains valid normalized coordinates
        format: class_id center_x center_y width height (0.0 to 1.0).
        """
        if not os.path.exists(label_path):
            return False, 0, "File does not exist"

        line_count = 0
        with open(label_path, "r") as f:
            for idx, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) != 5:
                    return False, line_count, f"Line {idx+1}: expected 5 elements, got {len(parts)}"
                cls_id, cx, cy, w, h = parts
                try:
                    cls_id = int(cls_id)
                    cx, cy, w, h = float(cx), float(cy), float(w), float(h)
                except ValueError:
                    return False, line_count, f"Line {idx+1}: non-numeric values found"

                if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                    return False, line_count, f"Line {idx+1}: coordinates outside [0.0, 1.0] range"
                line_count += 1

        return True, line_count, "Valid"

    def split_dataset(
        self,
        raw_images_dir: str,
        raw_labels_dir: str,
        train_ratio: float = 0.70,
        val_ratio: float = 0.20,
        test_ratio: float = 0.10,
        seed: int = 42
    ) -> Dict[str, int]:
        """
        Splits paired images and YOLO labels into train, val, and test subsets.
        """
        self.create_directory_structure()

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        images = [
            f for f in os.listdir(raw_images_dir)
            if os.path.splitext(f)[1].lower() in image_extensions
        ]

        random.seed(seed)
        random.shuffle(images)

        total = len(images)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        splits = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:]
        }

        counts = {}
        for split_name, img_list in splits.items():
            for img_file in img_list:
                base_name = os.path.splitext(img_file)[0]
                label_file = f"{base_name}.txt"

                src_img = os.path.join(raw_images_dir, img_file)
                src_lbl = os.path.join(raw_labels_dir, label_file)

                dst_img = os.path.join(self.target_dir, "images", split_name, img_file)
                dst_lbl = os.path.join(self.target_dir, "labels", split_name, label_file)

                shutil.copy2(src_img, dst_img)
                if os.path.exists(src_lbl):
                    shutil.copy2(src_lbl, dst_lbl)
                else:
                    # Create empty label file for background / negative samples
                    open(dst_lbl, "w").close()

            counts[split_name] = len(img_list)
            logger.info(f"Split '{split_name}': {len(img_list)} images processed.")

        return counts

    def generate_statistics(self) -> Dict[str, Any]:
        """Scans dataset directory and compiles annotation metrics."""
        stats: Dict[str, Any] = {"splits": {}, "class_distribution": {cls: 0 for cls in self.classes}}

        for split in ["train", "val", "test"]:
            lbl_dir = os.path.join(self.target_dir, "labels", split)
            img_dir = os.path.join(self.target_dir, "images", split)

            if not os.path.exists(lbl_dir):
                continue

            num_images = len(os.listdir(img_dir)) if os.path.exists(img_dir) else 0
            labels = os.listdir(lbl_dir)
            total_boxes = 0

            for lbl in labels:
                lbl_path = os.path.join(lbl_dir, lbl)
                with open(lbl_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            cls_id = int(parts[0])
                            if cls_id < len(self.classes):
                                cls_name = self.classes[cls_id]
                                stats["class_distribution"][cls_name] += 1
                            total_boxes += 1

            stats["splits"][split] = {
                "images": num_images,
                "annotations": total_boxes
            }

        logger.info(f"Dataset Statistics Summary: {json.dumps(stats, indent=2)}")
        return stats

if __name__ == "__main__":
    preparer = DatasetPreparer()
    preparer.create_directory_structure()
    print("Dataset directory prepared at ml/datasets/helmet_dataset")

