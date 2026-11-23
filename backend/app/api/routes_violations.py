import io
import csv
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy.orm import Session
from sqlalchemy import desc

from backend.app.database.session import get_db
from backend.app.models.violation import Violation
from backend.app.schemas.violation import ViolationOut, ViolationStatusUpdate, ViolationListResponse

router = APIRouter(prefix="/violations", tags=["Violation Records"])

@router.get("", response_model=ViolationListResponse)
def list_violations(
    violation_type: Optional[str] = Query(None, description="Filter by no_helmet, multiple_riders, wrong_way"),
    camera_id: Optional[str] = Query(None, description="Filter by camera ID"),
    review_status: Optional[str] = Query(None, description="Filter by pending, confirmed, rejected"),
    min_confidence: Optional[float] = Query(None, ge=0.0, le=1.0),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db)
):
    query = db.query(Violation)

    if violation_type:
        query = query.filter(Violation.violation_type == violation_type)
    if camera_id:
        query = query.filter(Violation.camera_id == camera_id)
    if review_status:
        query = query.filter(Violation.review_status == review_status)
    if min_confidence is not None:
        query = query.filter(Violation.violation_confidence >= min_confidence)

    total = query.count()
    offset = (page - 1) * page_size
    items = query.order_by(desc(Violation.created_at)).offset(offset).limit(page_size).all()

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": items
    }

@router.get("/export")
def export_violations(
    format: str = Query("csv", pattern="^(csv|json)$"),
    violation_type: Optional[str] = None,
    review_status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    query = db.query(Violation)
    if violation_type:
        query = query.filter(Violation.violation_type == violation_type)
    if review_status:
        query = query.filter(Violation.review_status == review_status)
    
    violations = query.order_by(desc(Violation.created_at)).all()

    if format == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "Violation ID", "Job ID", "Camera ID", "Tracking ID", "Violation Type",
            "Timestamp (s)", "Detection Conf", "Violation Conf", "Review Status",
            "Explanation", "Evidence Path", "Created At"
        ])
        for v in violations:
            writer.writerow([
                v.id, v.job_id, v.camera_id, v.tracking_id, v.violation_type,
                v.timestamp, v.detection_confidence, v.violation_confidence, v.review_status,
                v.explanation, v.annotated_evidence_path, v.created_at.isoformat()
            ])
        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=traffic_violations_export.csv"}
        )
    else:
        # JSON export
        export_data = [
            {
                "id": v.id,
                "job_id": v.job_id,
                "camera_id": v.camera_id,
                "tracking_id": v.tracking_id,
                "violation_type": v.violation_type,
                "timestamp": v.timestamp,
                "detection_confidence": v.detection_confidence,
                "violation_confidence": v.violation_confidence,
                "evidence_path": v.evidence_path,
                "annotated_evidence_path": v.annotated_evidence_path,
                "crop_path": v.crop_path,
                "review_status": v.review_status,
                "explanation": v.explanation,
                "created_at": v.created_at.isoformat()
            }
            for v in violations
        ]
        return export_data

@router.get("/{violation_id}", response_model=ViolationOut)
def get_violation(violation_id: str, db: Session = Depends(get_db)):
    viol = db.query(Violation).filter(Violation.id == violation_id).first()
    if not viol:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Violation '{violation_id}' not found."
        )
    return viol

@router.patch("/{violation_id}/status", response_model=ViolationOut)
def update_violation_status(
    violation_id: str,
    status_update: ViolationStatusUpdate,
    db: Session = Depends(get_db)
):
    viol = db.query(Violation).filter(Violation.id == violation_id).first()
    if not viol:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Violation '{violation_id}' not found."
        )
    viol.review_status = status_update.review_status
    db.commit()
    db.refresh(viol)
    return viol
