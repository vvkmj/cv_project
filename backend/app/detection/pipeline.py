import os
import uuid
import time
import logging
from typing import Dict, Any, List, Optional, Callable
import cv2
import numpy as np

from backend.app.detection.vehicle_detector import VehicleDetector
from backend.app.tracking.tracker import MultiObjectTracker
from backend.app.violations.association import RiderAssociator
from backend.app.violations.helmet_detector import HelmetDetector
from backend.app.violations.passenger_counter import PassengerCounter
from backend.app.violations.wrong_way import WrongWayDetector
from backend.app.violations.evidence_generator import EvidenceGenerator

logger = logging.getLogger(__name__)

class TrafficAnalysisPipeline:
    """
    Unified Computer Vision Engine orchestrating detection, tracking,
    rider association, violation verification, and evidence synthesis.
    """

    def __init__(
        self,
        vehicle_model_path: str = "yolov8n.pt",
        helmet_model_path: str = "ml/weights/helmet_yolov8.pt",
        confidence_threshold: float = 0.35,
        evidence_dir: str = "evidence",
        device: Optional[str] = None
    ):
        self.evidence_generator = EvidenceGenerator(base_dir=evidence_dir)
        self.vehicle_detector = VehicleDetector(
            model_path=vehicle_model_path,
            confidence_threshold=confidence_threshold,
            device=device
        )
        self.helmet_detector = HelmetDetector(
            weights_path=helmet_model_path,
            device=device
        )
        self.rider_associator = RiderAssociator()
        self.evidence_dir = evidence_dir

    def get_model_status(self) -> Dict[str, Any]:
        """Returns hardware device, loaded detection models, and helmet status."""
        v_status = self.vehicle_detector.get_status()
        h_status = self.helmet_detector.get_status()
        return {
            "vehicle_detector": v_status,
            "helmet_detector": h_status,
            "hardware_device": v_status.get("device", "cpu"),
            "pipeline_operational": True
        }

    def analyze_image(
        self,
        image_path: str,
        camera_config: Dict[str, Any],
        job_id: str
    ) -> Dict[str, Any]:
        """
        Processes a single static image:
        - Detects vehicles and pedestrians
        - Associates riders with motorcycles
        - Evaluates helmet violations (if model available) and passenger count
        - Generates evidentiary keyframe crops
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read image at {image_path}")

        h, w = image.shape[:2]
        cam_id = camera_config.get("id", "CAM-01")
        passenger_limit = camera_config.get("passenger_limit", 2)
        enabled_violations = camera_config.get("enabled_violations", {
            "no_helmet": True,
            "multiple_riders": True,
            "wrong_way": False # wrong-way requires temporal video tracking
        })

        # 1. Detect objects
        raw_detections = self.vehicle_detector.detect(image, frame_number=0, timestamp=0.0)

        # 2. Track / Assign temporary IDs for static image
        for idx, det in enumerate(raw_detections):
            det["tracking_id"] = idx + 1

        # 3. Associate riders
        motorcycles, persons = self.rider_associator.associate_riders(raw_detections, (h, w))

        violations: List[Dict[str, Any]] = []

        # 4. Check Multiple Riders & Helmet violations
        for moto in motorcycles:
            riders = moto.get("associated_riders", [])
            moto_track_id = moto.get("tracking_id", 1)

            # Multiple riders check
            if enabled_violations.get("multiple_riders", True) and len(riders) > passenger_limit:
                v_id = f"VIOL-{uuid.uuid4().hex[:8].upper()}"
                explanation = (
                    f"Motorcycle carrying {len(riders)} occupants, exceeding permitted limit of {passenger_limit}."
                )
                evidence_paths = self.evidence_generator.save_evidence(
                    violation_id=v_id,
                    original_frame=image,
                    annotated_frame=self.evidence_generator.draw_annotations(
                        image, raw_detections, [{"violation_type": "multiple_riders", "tracking_id": moto_track_id}],
                        frame_idx=0, timestamp=0.0, camera_id=cam_id
                    ),
                    crop_box=moto["bounding_box"]
                )

                violations.append({
                    "violation_id": v_id,
                    "job_id": job_id,
                    "camera_id": cam_id,
                    "tracking_id": moto_track_id,
                    "violation_type": "multiple_riders",
                    "timestamp": 0.0,
                    "detection_confidence": moto.get("confidence", 0.85),
                    "violation_confidence": 0.90,
                    "evidence_path": evidence_paths["evidence_path"],
                    "annotated_evidence_path": evidence_paths["annotated_evidence_path"],
                    "crop_path": evidence_paths["crop_path"],
                    "explanation": explanation,
                    "review_status": "pending"
                })

            # Helmet check on each rider
            if enabled_violations.get("no_helmet", True):
                for rider in riders:
                    h_res = self.helmet_detector.analyze_head_region(image, rider["head_box"])
                    rider["helmet_status"] = h_res["status"]
                    rider["helmet_confidence"] = h_res["confidence"]

                    # Only flag violation if explicitly classified as no_helmet with sufficient confidence
                    if h_res["status"] == "no_helmet" and h_res["confidence"] >= self.helmet_detector.confidence_threshold:
                        v_id = f"VIOL-{uuid.uuid4().hex[:8].upper()}"
                        r_track = rider.get("person_tracking_id", moto_track_id)
                        explanation = (
                            f"Motorcycle rider (Track #{r_track}) classified as NO HELMET with confidence {h_res['confidence']:.2f}."
                        )
                        evidence_paths = self.evidence_generator.save_evidence(
                            violation_id=v_id,
                            original_frame=image,
                            annotated_frame=self.evidence_generator.draw_annotations(
                                image, raw_detections, [{"violation_type": "no_helmet", "tracking_id": r_track}],
                                frame_idx=0, timestamp=0.0, camera_id=cam_id
                            ),
                            crop_box=rider["person_box"]
                        )

                        violations.append({
                            "violation_id": v_id,
                            "job_id": job_id,
                            "camera_id": cam_id,
                            "tracking_id": r_track,
                            "violation_type": "no_helmet",
                            "timestamp": 0.0,
                            "detection_confidence": rider.get("person_confidence", 0.85),
                            "violation_confidence": h_res["confidence"],
                            "evidence_path": evidence_paths["evidence_path"],
                            "annotated_evidence_path": evidence_paths["annotated_evidence_path"],
                            "crop_path": evidence_paths["crop_path"],
                            "explanation": explanation,
                            "review_status": "pending"
                        })

        # Generate final annotated overview
        annotated_full = self.evidence_generator.draw_annotations(
            image, raw_detections, violations, frame_idx=0, timestamp=0.0, camera_id=cam_id
        )
        overview_path = os.path.join(self.evidence_generator.annotated_dir, f"{job_id}_overview.jpg")
        cv2.imwrite(overview_path, annotated_full)

        return {
            "job_id": job_id,
            "media_type": "image",
            "total_frames": 1,
            "processed_frames": 1,
            "detections": raw_detections,
            "violations": violations,
            "annotated_media_path": overview_path,
            "summary": {
                "total_vehicles_detected": len(raw_detections),
                "violations_count": len(violations),
                "no_helmet_count": sum(1 for v in violations if v["violation_type"] == "no_helmet"),
                "multiple_riders_count": sum(1 for v in violations if v["violation_type"] == "multiple_riders"),
                "wrong_way_count": 0
            }
        }

    def analyze_video(
        self,
        video_path: str,
        camera_config: Dict[str, Any],
        job_id: str,
        progress_callback: Optional[Callable[[float, int, int], None]] = None
    ) -> Dict[str, Any]:
        """
        Processes video stream frame-by-frame:
        1. Object detection & multi-object tracking across frames.
        2. Rider-to-motorcycle association with head box extraction.
        3. Dedicated helmet analysis (or honest unknown/unavailable reporting).
        4. Temporal multiple-passenger counting with stabilization.
        5. Wrong-way trajectory analysis against camera-configured expected direction.
        6. Deduplication and evidence synthesis (images + annotated video).
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found at {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file at {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

        cam_id = camera_config.get("id", "CAM-01")
        passenger_limit = camera_config.get("passenger_limit", 2)
        expected_dir = camera_config.get("expected_direction", {"angle_deg": 90.0})
        enabled_violations = camera_config.get("enabled_violations", {
            "no_helmet": True,
            "multiple_riders": True,
            "wrong_way": True
        })

        # Initialize trackers & analyzers
        tracker = MultiObjectTracker(iou_threshold=0.25, max_age=15)
        passenger_counter = PassengerCounter(buffer_size=5, min_consecutive_frames=3)
        wrong_way_detector = WrongWayDetector(min_movement_pixels=25.0, consecutive_frames_required=3)
        alerted_helmet_tracks = set()

        # Video writer for annotated output
        annotated_video_path = os.path.join(self.evidence_generator.video_dir, f"{job_id}_annotated.mp4")
        # Try mp4v or avc1
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(annotated_video_path, fourcc, fps, (width, height))

        all_detections: List[Dict[str, Any]] = []
        violations: List[Dict[str, Any]] = []
        unique_vehicle_ids = set()

        frame_idx = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            timestamp = frame_idx / fps

            # 1. Detection
            raw_dets = self.vehicle_detector.detect(frame, frame_number=frame_idx, timestamp=timestamp)

            # 2. Tracking
            tracked_dets = tracker.update(raw_dets)
            all_detections.extend(tracked_dets)

            for d in tracked_dets:
                if d.get("tracking_id"):
                    unique_vehicle_ids.add(d["tracking_id"])

            active_frame_violations = []

            # 3. Rider Association
            motorcycles, persons = self.rider_associator.associate_riders(tracked_dets, (height, width))

            # 4. Check Multiple Riders & Helmet violations
            for moto in motorcycles:
                m_track_id = moto.get("tracking_id")
                riders = moto.get("associated_riders", [])

                # Passenger count check
                if enabled_violations.get("multiple_riders", True):
                    p_viol = passenger_counter.update(m_track_id, riders, passenger_limit=passenger_limit)
                    if p_viol:
                        v_id = f"VIOL-{uuid.uuid4().hex[:8].upper()}"
                        annot_frame = self.evidence_generator.draw_annotations(
                            frame, tracked_dets, [p_viol], frame_idx=frame_idx, timestamp=timestamp, camera_id=cam_id
                        )
                        paths = self.evidence_generator.save_evidence(
                            violation_id=v_id,
                            original_frame=frame,
                            annotated_frame=annot_frame,
                            crop_box=moto["bounding_box"]
                        )
                        viol_record = {
                            "violation_id": v_id,
                            "job_id": job_id,
                            "camera_id": cam_id,
                            "tracking_id": m_track_id,
                            "violation_type": "multiple_riders",
                            "timestamp": round(timestamp, 2),
                            "detection_confidence": moto.get("confidence", 0.85),
                            "violation_confidence": p_viol.get("confidence", 0.90),
                            "evidence_path": paths["evidence_path"],
                            "annotated_evidence_path": paths["annotated_evidence_path"],
                            "crop_path": paths["crop_path"],
                            "explanation": p_viol["explanation"],
                            "review_status": "pending"
                        }
                        violations.append(viol_record)
                        active_frame_violations.append(viol_record)

                # Helmet check
                if enabled_violations.get("no_helmet", True):
                    for rider in riders:
                        h_res = self.helmet_detector.analyze_head_region(frame, rider["head_box"])
                        r_track = rider.get("person_tracking_id", m_track_id)
                        if h_res["status"] == "no_helmet" and h_res["confidence"] >= self.helmet_detector.confidence_threshold:
                            if r_track not in alerted_helmet_tracks:
                                alerted_helmet_tracks.add(r_track)
                                v_id = f"VIOL-{uuid.uuid4().hex[:8].upper()}"
                                annot_frame = self.evidence_generator.draw_annotations(
                                    frame, tracked_dets, [{"violation_type": "no_helmet", "tracking_id": r_track}],
                                    frame_idx=frame_idx, timestamp=timestamp, camera_id=cam_id
                                )
                                paths = self.evidence_generator.save_evidence(
                                    violation_id=v_id,
                                    original_frame=frame,
                                    annotated_frame=annot_frame,
                                    crop_box=rider["person_box"]
                                )
                                viol_record = {
                                    "violation_id": v_id,
                                    "job_id": job_id,
                                    "camera_id": cam_id,
                                    "tracking_id": r_track,
                                    "violation_type": "no_helmet",
                                    "timestamp": round(timestamp, 2),
                                    "detection_confidence": rider.get("person_confidence", 0.85),
                                    "violation_confidence": h_res["confidence"],
                                    "evidence_path": paths["evidence_path"],
                                    "annotated_evidence_path": paths["annotated_evidence_path"],
                                    "crop_path": paths["crop_path"],
                                    "explanation": f"Rider on Track #{m_track_id} detected without helmet with confidence {h_res['confidence']:.2f}.",
                                    "review_status": "pending"
                                }
                                violations.append(viol_record)
                                active_frame_violations.append(viol_record)

            # 5. Wrong-Way Driving Analysis
            if enabled_violations.get("wrong_way", True):
                for track in tracker.tracks:
                    # Exclude pedestrians from wrong-way driving rules
                    if track.object_class == "person":
                        continue
                    ww_viol = wrong_way_detector.evaluate_track(track, expected_dir)
                    if ww_viol:
                        v_id = f"VIOL-{uuid.uuid4().hex[:8].upper()}"
                        annot_frame = self.evidence_generator.draw_annotations(
                            frame, tracked_dets, [ww_viol], frame_idx=frame_idx, timestamp=timestamp, camera_id=cam_id
                        )
                        paths = self.evidence_generator.save_evidence(
                            violation_id=v_id,
                            original_frame=frame,
                            annotated_frame=annot_frame,
                            crop_box={
                                "x1": track.bbox[0],
                                "y1": track.bbox[1],
                                "x2": track.bbox[2],
                                "y2": track.bbox[3]
                            }
                        )
                        viol_record = {
                            "violation_id": v_id,
                            "job_id": job_id,
                            "camera_id": cam_id,
                            "tracking_id": track.track_id,
                            "violation_type": "wrong_way",
                            "timestamp": round(timestamp, 2),
                            "detection_confidence": track.confidence,
                            "violation_confidence": ww_viol["confidence"],
                            "evidence_path": paths["evidence_path"],
                            "annotated_evidence_path": paths["annotated_evidence_path"],
                            "crop_path": paths["crop_path"],
                            "explanation": ww_viol["explanation"],
                            "review_status": "pending"
                        }
                        violations.append(viol_record)
                        active_frame_violations.append(viol_record)

            # Render HUD and write frame to annotated video
            annotated_frame = self.evidence_generator.draw_annotations(
                frame, tracked_dets, active_frame_violations,
                frame_idx=frame_idx, timestamp=timestamp, camera_id=cam_id
            )
            out_writer.write(annotated_frame)

            frame_idx += 1
            if progress_callback and (frame_idx % 5 == 0 or frame_idx == total_frames):
                pct = round(min(100.0, (frame_idx / total_frames) * 100.0), 1)
                progress_callback(pct, frame_idx, total_frames)

        cap.release()
        out_writer.release()

        elapsed = time.time() - start_time
        fps_achieved = round(frame_idx / max(1e-5, elapsed), 2)
        logger.info(f"Processed {frame_idx} video frames in {elapsed:.2f}s ({fps_achieved} FPS)")

        return {
            "job_id": job_id,
            "media_type": "video",
            "total_frames": total_frames,
            "processed_frames": frame_idx,
            "processing_fps": fps_achieved,
            "annotated_video_path": annotated_video_path,
            "detections_count": len(all_detections),
            "unique_vehicles": len(unique_vehicle_ids),
            "violations": violations,
            "summary": {
                "total_vehicles_detected": len(unique_vehicle_ids),
                "violations_count": len(violations),
                "no_helmet_count": sum(1 for v in violations if v["violation_type"] == "no_helmet"),
                "multiple_riders_count": sum(1 for v in violations if v["violation_type"] == "multiple_riders"),
                "wrong_way_count": sum(1 for v in violations if v["violation_type"] == "wrong_way")
            }
        }

