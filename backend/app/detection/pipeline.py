import os
import uuid
import time
import logging
import math
from contextlib import ExitStack
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


class HelmetObservations:
    def __init__(self):
        self.streaks = {}
        self.reported = set()

    def observe(self, rider_id, motorcycle_id, frame_number, timestamp, status):
        if rider_id is None or motorcycle_id is None or rider_id in self.reported:
            return False
        previous = self.streaks.pop(rider_id, None)
        if status != "no_helmet":
            return False
        count = 1
        if previous and previous[0] == motorcycle_id and frame_number - previous[1] == 1 and timestamp - previous[2] <= 0.5:
            count += previous[3]
        if count >= 3:
            self.reported.add(rider_id)
            return True
        self.streaks[rider_id] = (motorcycle_id, frame_number, timestamp, count)
        return False

    def prune(self, active_ids):
        self.reported.intersection_update(active_ids)
        self.streaks = {key: state for key, state in self.streaks.items() if key in active_ids}


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
        device: Optional[str] = None,
        helmet_confidence: float = 0.50,
    ):
        self.evidence_generator = EvidenceGenerator(base_dir=evidence_dir)
        self.vehicle_detector = VehicleDetector(
            model_path=vehicle_model_path,
            confidence_threshold=confidence_threshold,
            device=device
        )
        self.helmet_detector = HelmetDetector(
            weights_path=helmet_model_path,
            device=self.vehicle_detector.device,
            confidence_threshold=helmet_confidence,
        )
        self.rider_associator = RiderAssociator()
        self.evidence_dir = evidence_dir

    def _helmet_results(self, image, motorcycles, persons, excluded=()):
        riders = [rider for moto in motorcycles for rider in moto["associated_riders"]
                  if rider["person_tracking_id"] not in excluded]
        results = self.helmet_detector.analyze_heads(
            image, [rider["head_box"] for rider in riders], [person["head_box"] for person in persons]
        )
        return {rider["person_tracking_id"]: result for rider, result in zip(riders, results, strict=True)}

    @staticmethod
    def _combined_crop(motorcycle, rider):
        bike, person = motorcycle["bounding_box"], rider["person_box"]
        return {key: (min if key.endswith("1") else max)(bike[key], person[key]) for key in bike}

    def get_model_status(self) -> Dict[str, Any]:
        """Returns hardware device, loaded detection models, and helmet status."""
        v_status = self.vehicle_detector.get_status()
        h_status = self.helmet_detector.get_status()
        return {
            "vehicle_detector": v_status,
            "helmet_detector": h_status,
            "hardware_device": v_status.get("device", "cpu"),
            "pipeline_operational": v_status["is_loaded"]
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
        helmet_results = self._helmet_results(image, motorcycles, persons) if enabled_violations.get("no_helmet", True) else {}

        violations: List[Dict[str, Any]] = []

        # 4. Check Multiple Riders & Helmet violations
        for moto in motorcycles:
            riders = moto.get("associated_riders", [])
            moto_track_id = moto.get("tracking_id", 1)

            # Multiple riders check
            if enabled_violations.get("multiple_riders", True) and len(riders) > passenger_limit:
                v_id = f"VIOL-{uuid.uuid4().hex.upper()}"
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
                    "frame_number": 0,
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
                    h_res = helmet_results[rider["person_tracking_id"]]
                    rider["helmet_status"] = h_res["status"]
                    rider["helmet_confidence"] = h_res["confidence"]

                    # Only flag violation if explicitly classified as no_helmet with sufficient confidence
                    if h_res["status"] == "no_helmet" and h_res["confidence"] >= self.helmet_detector.confidence_threshold:
                        v_id = f"VIOL-{uuid.uuid4().hex.upper()}"
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
                            crop_box=self._combined_crop(moto, rider)
                        )

                        violations.append({
                            "violation_id": v_id,
                            "job_id": job_id,
                            "camera_id": cam_id,
                            "tracking_id": r_track,
                            "violation_type": "no_helmet",
                            "timestamp": 0.0,
                            "frame_number": 0,
                            "motorcycle_tracking_id": moto_track_id,
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
        progress_callback: Optional[Callable[[float, int, int], None]] = None,
        incident_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        write_annotated_video: bool = True,
    ) -> Dict[str, Any]:
        with ExitStack() as resources:
            return self._analyze_video(video_path, camera_config, job_id, progress_callback, incident_callback,
                                       write_annotated_video, resources)

    def _analyze_video(self, video_path, camera_config, job_id, progress_callback, incident_callback,
                       write_annotated_video, resources):
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
        resources.callback(cap.release)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file at {video_path}")

        reported_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        total_frames = int(reported_count) if math.isfinite(reported_count) and reported_count > 0 else 0
        reported_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = reported_fps if math.isfinite(reported_fps) and reported_fps > 0 else 25.0

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
        helmet_observations = HelmetObservations()

        # Video writer for annotated output
        annotated_video_path = None
        out_writer = None

        detections_count = 0
        violations: List[Dict[str, Any]] = []
        previous_vehicles = set()
        unique_vehicles = 0

        frame_idx = 0
        start_time = time.perf_counter()
        inference_seconds = 0.0
        previous_timestamp = -1.0 / fps

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            height, width = frame.shape[:2]
            if write_annotated_video and out_writer is None:
                annotated_video_path = os.path.join(self.evidence_generator.video_dir, f"{job_id}_annotated.mp4")
                out_writer = cv2.VideoWriter(annotated_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
                resources.callback(out_writer.release)
                if not out_writer.isOpened():
                    raise RuntimeError("Could not initialize the annotated video encoder")

            decoded_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            timestamp = (
                decoded_timestamp if math.isfinite(decoded_timestamp) and decoded_timestamp >= 0
                and decoded_timestamp > previous_timestamp else previous_timestamp + 1.0 / fps
            )
            previous_timestamp = timestamp

            # 1. Detection
            inference_start = time.perf_counter()
            raw_dets = self.vehicle_detector.detect(frame, frame_number=frame_idx, timestamp=timestamp)
            inference_seconds += time.perf_counter() - inference_start

            # 2. Tracking
            tracked_dets = tracker.update(raw_dets)
            detections_count += len(tracked_dets)
            active_ids = {track.track_id for track in tracker.tracks}
            active_vehicles = {track.track_id for track in tracker.tracks if track.object_class != "person"}
            unique_vehicles += len(active_vehicles - previous_vehicles)
            previous_vehicles = active_vehicles
            helmet_observations.prune(active_ids)
            for state in (passenger_counter.count_history, wrong_way_detector.consecutive_wrong_way_counts):
                for expired in state.keys() - active_ids:
                    del state[expired]
            passenger_counter.reported_tracks.intersection_update(active_ids)
            wrong_way_detector.reported_tracks.intersection_update(active_ids)

            active_frame_violations = []

            # 3. Rider Association
            motorcycles, persons = self.rider_associator.associate_riders(tracked_dets, (height, width))
            inference_start = time.perf_counter()
            helmet_results = self._helmet_results(frame, motorcycles, persons, helmet_observations.reported) if enabled_violations.get("no_helmet", True) else {}
            inference_seconds += time.perf_counter() - inference_start

            # 4. Check Multiple Riders & Helmet violations
            for moto in motorcycles:
                m_track_id = moto.get("tracking_id")
                riders = moto.get("associated_riders", [])

                # Passenger count check
                if enabled_violations.get("multiple_riders", True):
                    p_viol = passenger_counter.update(m_track_id, riders, passenger_limit=passenger_limit)
                    if p_viol:
                        v_id = f"VIOL-{uuid.uuid4().hex.upper()}"
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
                            "frame_number": frame_idx,
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
                        r_track = rider.get("person_tracking_id", m_track_id)
                        h_res = helmet_results.get(r_track)
                        if not h_res or not helmet_observations.observe(r_track, m_track_id, frame_idx, timestamp, h_res["status"]):
                            continue
                        v_id = f"VIOL-{uuid.uuid4().hex.upper()}"
                        annot_frame = self.evidence_generator.draw_annotations(
                            frame, tracked_dets, [{"violation_type": "no_helmet", "tracking_id": r_track}],
                            frame_idx=frame_idx, timestamp=timestamp, camera_id=cam_id
                        )
                        paths = self.evidence_generator.save_evidence(
                            violation_id=v_id, original_frame=frame, annotated_frame=annot_frame,
                            crop_box=self._combined_crop(moto, rider),
                        )
                        viol_record = {
                            "violation_id": v_id, "job_id": job_id, "camera_id": cam_id, "tracking_id": r_track,
                            "violation_type": "no_helmet", "timestamp": timestamp, "frame_number": frame_idx,
                            "motorcycle_tracking_id": m_track_id, "detection_confidence": rider["person_confidence"],
                            "violation_confidence": h_res["confidence"], **paths,
                            "explanation": f"Three consecutive no-helmet observations on motorcycle track #{m_track_id}.",
                            "review_status": "pending",
                        }
                        violations.append(viol_record)
                        active_frame_violations.append(viol_record)

            # 5. Wrong-Way Driving Analysis
            if enabled_violations.get("wrong_way", True):
                for track in tracker.tracks:
                    # Exclude pedestrians from wrong-way driving rules
                    if track.object_class == "person" or track.time_since_update:
                        continue
                    ww_viol = wrong_way_detector.evaluate_track(track, expected_dir)
                    if ww_viol:
                        v_id = f"VIOL-{uuid.uuid4().hex.upper()}"
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
                            "frame_number": frame_idx,
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

            if incident_callback:
                for incident in active_frame_violations:
                    incident_callback(incident)
            if out_writer is not None:
                out_writer.write(self.evidence_generator.draw_annotations(
                    frame, tracked_dets, active_frame_violations, frame_idx=frame_idx, timestamp=timestamp, camera_id=cam_id
                ))

            frame_idx += 1
            if progress_callback and (frame_idx % 5 == 0 or frame_idx == total_frames):
                pct = round(min(100.0, (frame_idx / max(1, total_frames)) * 100.0), 1) if total_frames else 0.0
                progress_callback(pct, frame_idx, total_frames)

        if not frame_idx or (total_frames and frame_idx < total_frames - max(2, int(total_frames * 0.02))):
            raise ValueError("Video is empty or ended before the expected frame count")
        if progress_callback:
            progress_callback(100.0, frame_idx, max(frame_idx, total_frames))

        elapsed = time.perf_counter() - start_time
        fps_achieved = round(frame_idx / max(1e-5, elapsed), 2)
        logger.info(f"Processed {frame_idx} video frames in {elapsed:.2f}s ({fps_achieved} FPS)")

        return {
            "job_id": job_id,
            "media_type": "video",
            "total_frames": max(frame_idx, total_frames),
            "processed_frames": frame_idx,
            "processing_fps": fps_achieved,
            "mean_inference_ms": round(1000 * inference_seconds / frame_idx, 2),
            "annotated_video_path": annotated_video_path,
            "detections_count": detections_count,
            "unique_vehicles": unique_vehicles,
            "violations": violations,
            "summary": {
                "total_vehicles_detected": unique_vehicles,
                "violations_count": len(violations),
                "no_helmet_count": sum(1 for v in violations if v["violation_type"] == "no_helmet"),
                "multiple_riders_count": sum(1 for v in violations if v["violation_type"] == "multiple_riders"),
                "wrong_way_count": sum(1 for v in violations if v["violation_type"] == "wrong_way")
            }
        }

