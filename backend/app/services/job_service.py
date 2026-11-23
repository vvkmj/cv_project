import os
import time
from datetime import datetime
import logging
from threading import Lock
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session

from backend.app.database.session import SessionLocal
from backend.app.models.job import AnalysisJob
from backend.app.models.violation import Violation
from backend.app.models.detection import Detection
from backend.app.models.camera import Camera
from backend.app.detection.pipeline import TrafficAnalysisPipeline
from backend.app.core.config import settings

logger = logging.getLogger(__name__)

# Reusable pipeline instance
_pipeline: Optional[TrafficAnalysisPipeline] = None
_inference_lock = Lock()

def get_pipeline() -> TrafficAnalysisPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = TrafficAnalysisPipeline(
            vehicle_model_path=settings.VEHICLE_MODEL_PATH,
            helmet_model_path=settings.HELMET_MODEL_PATH,
            confidence_threshold=settings.DETECTION_CONFIDENCE,
            helmet_confidence=settings.HELMET_CONFIDENCE,
            evidence_dir=settings.EVIDENCE_DIR
        )
    return _pipeline


def process_image_job(job_id: str, file_path: str, camera_id: str):
    """Executes static image analysis and persists violations and detections."""
    db: Session = SessionLocal()
    try:
        job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
        if not job:
            logger.error(f"Job {job_id} not found in database.")
            return

        job.status = "processing"
        job.progress = 10.0
        db.commit()

        # Fetch camera configuration
        camera = db.query(Camera).filter(Camera.id == camera_id).first()
        camera_config = {
            "id": camera.id if camera else "CAM-01",
            "passenger_limit": camera.passenger_limit if camera else settings.DEFAULT_PASSENGER_LIMIT,
            "expected_direction": camera.expected_direction if camera else {"angle_deg": 90.0},
            "enabled_violations": camera.enabled_violations if camera else {"no_helmet": True, "multiple_riders": True, "wrong_way": False}
        }

        with _inference_lock:
            result = get_pipeline().analyze_image(file_path, camera_config, job_id)

        # Persist violations
        for v in result.get("violations", []):
            violation = Violation(
                id=v["violation_id"],
                job_id=job_id,
                camera_id=camera_config["id"],
                tracking_id=v.get("tracking_id"),
                violation_type=v["violation_type"],
                timestamp=v.get("timestamp", 0.0),
                detection_confidence=v.get("detection_confidence", 0.8),
                violation_confidence=v.get("violation_confidence", 0.8),
                evidence_path=v["evidence_path"],
                annotated_evidence_path=v["annotated_evidence_path"],
                crop_path=v.get("crop_path"),
                review_status="pending",
                explanation=v["explanation"],
                created_at=datetime.utcnow()
            )
            db.add(violation)

        # Persist detections
        for det in result.get("detections", []):
            detection_row = Detection(
                job_id=job_id,
                tracking_id=det.get("tracking_id"),
                frame_number=det.get("frame_number", 0),
                object_class=det["object_class"],
                confidence=det["confidence"],
                bounding_box=det["bounding_box"],
                timestamp=det.get("timestamp", 0.0)
            )
            db.add(detection_row)

        job.status = "completed"
        job.progress = 100.0
        job.total_frames = 1
        job.processed_frames = 1
        job.annotated_path = result.get("annotated_media_path")
        job.summary = result.get("summary")
        job.completed_at = datetime.utcnow()
        db.commit()
        logger.info(f"Image analysis job {job_id} successfully completed.")

    except Exception as e:
        logger.error(f"Error processing image job {job_id}: {e}", exc_info=True)
        job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
        if job:
            job.status = "failed"
            job.error_message = "Image analysis failed. Check the input and detector configuration."
            job.completed_at = datetime.utcnow()
            db.commit()
    finally:
        db.close()


def process_video_job(job_id: str, file_path: str, camera_id: str):
    """Executes asynchronous video analysis with progress callback updates."""
    db: Session = SessionLocal()
    try:
        job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
        if not job:
            logger.error(f"Job {job_id} not found.")
            return

        job.status = "processing"
        job.progress = 1.0
        db.commit()

        camera = db.query(Camera).filter(Camera.id == camera_id).first()
        camera_config = {
            "id": camera.id if camera else "CAM-01",
            "passenger_limit": camera.passenger_limit if camera else settings.DEFAULT_PASSENGER_LIMIT,
            "expected_direction": camera.expected_direction if camera else {"angle_deg": 90.0},
            "enabled_violations": camera.enabled_violations if camera else {"no_helmet": True, "multiple_riders": True, "wrong_way": True}
        }

        def progress_callback(pct: float, current_frame: int, total_frames: int):
            try:
                # Open isolated session for progress update to avoid long locks
                p_db = SessionLocal()
                p_job = p_db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
                if p_job:
                    p_job.progress = pct
                    p_job.processed_frames = current_frame
                    p_job.total_frames = total_frames
                    p_db.commit()
                p_db.close()
            except Exception as pe:
                logger.debug(f"Progress update warning: {pe}")

        with _inference_lock:
            result = get_pipeline().analyze_video(file_path, camera_config, job_id, progress_callback=progress_callback)

        # Refresh job
        job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()

        # Persist violations
        for v in result.get("violations", []):
            violation = Violation(
                id=v["violation_id"],
                job_id=job_id,
                camera_id=camera_config["id"],
                tracking_id=v.get("tracking_id"),
                violation_type=v["violation_type"],
                timestamp=v.get("timestamp", 0.0),
                detection_confidence=v.get("detection_confidence", 0.8),
                violation_confidence=v.get("violation_confidence", 0.8),
                evidence_path=v["evidence_path"],
                annotated_evidence_path=v["annotated_evidence_path"],
                crop_path=v.get("crop_path"),
                review_status="pending",
                explanation=v["explanation"],
                created_at=datetime.utcnow()
            )
            db.add(violation)

        job.status = "completed"
        job.progress = 100.0
        job.total_frames = result.get("total_frames", 1)
        job.processed_frames = result.get("processed_frames", 1)
        job.annotated_path = result.get("annotated_video_path")
        job.summary = result.get("summary")
        job.completed_at = datetime.utcnow()
        db.commit()
        logger.info(f"Video analysis job {job_id} successfully completed.")

    except Exception as e:
        logger.error(f"Error processing video job {job_id}: {e}", exc_info=True)
        job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
        if job:
            job.status = "failed"
            job.error_message = "Video analysis failed. Check the input and detector configuration."
            job.completed_at = datetime.utcnow()
            db.commit()
    finally:
        db.close()

