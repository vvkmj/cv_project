import os
import uuid
import shutil
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.app.database.session import get_db
from backend.app.models.job import AnalysisJob
from backend.app.models.violation import Violation
from backend.app.models.detection import Detection
from backend.app.schemas.job import AnalysisJobOut, AnalysisJobDetail
from backend.app.services.job_service import process_image_job, process_video_job
from backend.app.core.config import settings

router = APIRouter(prefix="", tags=["Media Processing & Analysis"])

ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

@router.post("/media/upload")
async def upload_media(file: UploadFile = File(...)):
    filename = file.filename or "media_upload"
    ext = os.path.splitext(filename)[1].lower()

    if ext not in ALLOWED_IMAGE_EXTS and ext not in ALLOWED_VIDEO_EXTS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Supported: {list(ALLOWED_IMAGE_EXTS | ALLOWED_VIDEO_EXTS)}"
        )

    media_type = "image" if ext in ALLOWED_IMAGE_EXTS else "video"
    unique_name = f"{uuid.uuid4().hex[:10]}_{filename}"
    target_path = os.path.join(settings.UPLOAD_DIR, unique_name)

    with open(target_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "filename": unique_name,
        "original_filename": filename,
        "media_type": media_type,
        "file_path": target_path,
        "size_bytes": os.path.getsize(target_path)
    }

@router.post("/analysis/image", response_model=AnalysisJobDetail)
def analyze_image_endpoint(
    filename: str = Form(...),
    camera_id: str = Form("CAM-01"),
    db: Session = Depends(get_db)
):
    file_path = os.path.join(settings.UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        # Also check sample_media
        sample_path = os.path.join(settings.SAMPLE_MEDIA_DIR, filename)
        if os.path.exists(sample_path):
            file_path = sample_path
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File '{filename}' not found on server."
            )

    job_id = f"JOB-IMG-{uuid.uuid4().hex[:8].upper()}"
    new_job = AnalysisJob(
        id=job_id,
        filename=filename,
        media_type="image",
        camera_id=camera_id,
        status="processing",
        progress=10.0
    )
    db.add(new_job)
    db.commit()

    # Run image analysis synchronously as it takes < 1 second
    process_image_job(job_id, file_path, camera_id)

    db.refresh(new_job)
    violations = db.query(Violation).filter(Violation.job_id == job_id).all()
    detections = db.query(Detection).filter(Detection.job_id == job_id).all()

    return AnalysisJobDetail(
        id=new_job.id,
        filename=new_job.filename,
        media_type=new_job.media_type,
        camera_id=new_job.camera_id,
        status=new_job.status,
        progress=new_job.progress,
        total_frames=new_job.total_frames,
        processed_frames=new_job.processed_frames,
        summary=new_job.summary,
        annotated_path=new_job.annotated_path,
        error_message=new_job.error_message,
        created_at=new_job.created_at,
        completed_at=new_job.completed_at,
        violations=violations,
        detections=detections
    )

@router.post("/analysis/video", response_model=AnalysisJobOut, status_code=status.HTTP_202_ACCEPTED)
def analyze_video_endpoint(
    background_tasks: BackgroundTasks,
    filename: str = Form(...),
    camera_id: str = Form("CAM-01"),
    db: Session = Depends(get_db)
):
    file_path = os.path.join(settings.UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        sample_path = os.path.join(settings.SAMPLE_MEDIA_DIR, filename)
        if os.path.exists(sample_path):
            file_path = sample_path
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File '{filename}' not found."
            )

    job_id = f"JOB-VID-{uuid.uuid4().hex[:8].upper()}"
    new_job = AnalysisJob(
        id=job_id,
        filename=filename,
        media_type="video",
        camera_id=camera_id,
        status="queued",
        progress=0.0
    )
    db.add(new_job)
    db.commit()
    db.refresh(new_job)

    # Dispatch to background task worker to prevent blocking HTTP client
    background_tasks.add_task(process_video_job, job_id, file_path, camera_id)

    return new_job

@router.get("/analysis/{job_id}", response_model=AnalysisJobOut)
def get_job_status(job_id: str, db: Session = Depends(get_db)):
    job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis job '{job_id}' not found."
        )
    return job

@router.get("/analysis/{job_id}/results", response_model=AnalysisJobDetail)
def get_job_results(job_id: str, db: Session = Depends(get_db)):
    job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis job '{job_id}' not found."
        )

    violations = db.query(Violation).filter(Violation.job_id == job_id).all()
    detections = db.query(Detection).filter(Detection.job_id == job_id).all()

    return AnalysisJobDetail(
        id=job.id,
        filename=job.filename,
        media_type=job.media_type,
        camera_id=job.camera_id,
        status=job.status,
        progress=job.progress,
        total_frames=job.total_frames,
        processed_frames=job.processed_frames,
        summary=job.summary,
        annotated_path=job.annotated_path,
        error_message=job.error_message,
        created_at=job.created_at,
        completed_at=job.completed_at,
        violations=violations,
        detections=detections
    )

@router.get("/analysis/{job_id}/annotated-video")
def get_annotated_video(job_id: str, db: Session = Depends(get_db)):
    job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis job '{job_id}' not found."
        )

    if not job.annotated_path or not os.path.exists(job.annotated_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Annotated video file is not available or still processing."
        )

    return FileResponse(
        job.annotated_path,
        media_type="video/mp4",
        filename=os.path.basename(job.annotated_path)
    )

