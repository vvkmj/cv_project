import os
import psutil
from datetime import datetime
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.database.session import get_db
from backend.app.models.job import AnalysisJob
from backend.app.models.violation import Violation
from backend.app.services.job_service import get_pipeline
from backend.app.core.config import settings

router = APIRouter(tags=["Health & System"])

@router.get("/health")
def get_health():
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/system/status")
def get_system_status(db: Session = Depends(get_db)):
    # Hardware stats
    cpu_percent = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage(".")

    # DB metrics
    active_jobs = db.query(AnalysisJob).filter(AnalysisJob.status == "processing").count()
    completed_jobs = db.query(AnalysisJob).filter(AnalysisJob.status == "completed").count()
    total_violations = db.query(Violation).count()

    return {
        "system": {
            "cpu_usage_pct": cpu_percent,
            "memory_usage_pct": mem.percent,
            "memory_available_mb": round(mem.available / (1024 * 1024), 1),
            "disk_free_gb": round(disk.free / (1024 * 1024 * 1024), 2)
        },
        "workload": {
            "active_processing_jobs": active_jobs,
            "completed_jobs": completed_jobs,
            "total_violations_recorded": total_violations
        },
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/models/status")
def get_model_status():
    pipeline = get_pipeline()
    return pipeline.get_model_status()

