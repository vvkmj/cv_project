import os
import time
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.database.session import SessionLocal, Base, engine
from backend.app.services.job_service import process_video_job
from backend.app.models.job import AnalysisJob
from backend.app.models.violation import Violation

client = TestClient(app)

def test_full_video_processing_end_to_end():
    """
    End-to-End Workflow Verification:
    1. Verify video source exists
    2. Register video analysis job
    3. Execute video analysis pipeline
    4. Confirm detections, tracking continuity, and violation events
    5. Check evidence generation on disk (annotated video, frames, crops)
    6. Verify operator review status cycle
    """
    sample_vid = "traffic_video.mp4"
    sample_vid_path = os.path.join("sample_media", sample_vid)
    assert os.path.exists(sample_vid_path), f"Sample video must exist at {sample_vid_path}"

    job_id = "JOB-TEST-E2E-001"
    db = SessionLocal()
    # Clean previous run if present
    db.query(Violation).filter(Violation.job_id == job_id).delete()
    db.query(AnalysisJob).filter(AnalysisJob.id == job_id).delete()
    db.commit()

    # Create job record
    new_job = AnalysisJob(
        id=job_id,
        filename=sample_vid,
        media_type="video",
        camera_id="CAM-01",
        status="processing",
        progress=0.0
    )
    db.add(new_job)
    db.commit()
    db.close()

    # Execute processing
    process_video_job(job_id, sample_vid_path, "CAM-01")

    # Verify Job Completed
    res = client.get(f"/api/analysis/{job_id}/results")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "completed"
    assert data["processed_frames"] > 0
    assert data["progress"] == 100.0

    # Verify Annotated Video generated on disk
    annotated_video_path = data.get("annotated_path")
    assert annotated_video_path is not None
    assert os.path.exists(annotated_video_path)
    assert os.path.getsize(annotated_video_path) > 0

    # Verify Violations persisted
    violations = data["violations"]
    assert len(violations) > 0, "Video with opposing vehicle must yield wrong-way or multiple-rider violations"

    # Verify evidence files exist on disk for each violation
    for v in violations:
        assert os.path.exists(v["evidence_path"]), f"Original evidence missing: {v['evidence_path']}"
        assert os.path.exists(v["annotated_evidence_path"]), f"Annotated evidence missing: {v['annotated_evidence_path']}"
        if v.get("crop_path"):
            assert os.path.exists(v["crop_path"]), f"Crop evidence missing: {v['crop_path']}"

    # Verify Review Status workflow
    viol_id = violations[0]["id"]
    patch_res = client.patch(f"/api/violations/{viol_id}/status", json={"review_status": "confirmed"})
    assert patch_res.status_code == 200
    assert patch_res.json()["review_status"] == "confirmed"

