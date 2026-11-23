from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from backend.app.schemas.violation import ViolationOut
from backend.app.schemas.detection import DetectionOut

class AnalysisJobCreate(BaseModel):
    filename: str
    camera_id: str = "CAM-01"
    confidence_threshold: Optional[float] = 0.35
    passenger_limit: Optional[int] = 2

class AnalysisJobOut(BaseModel):
    id: str
    filename: str
    media_type: str
    camera_id: str
    status: str
    progress: float
    total_frames: int
    processed_frames: int
    summary: Optional[Dict[str, Any]] = None
    annotated_path: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class AnalysisJobDetail(AnalysisJobOut):
    violations: List[ViolationOut] = []
    detections: List[DetectionOut] = []

