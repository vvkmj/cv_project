from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class ViolationStatusUpdate(BaseModel):
    review_status: str = Field(..., pattern="^(pending|confirmed|rejected)$")

class ViolationOut(BaseModel):
    id: str
    job_id: str
    camera_id: str
    tracking_id: Optional[int] = None
    violation_type: str
    timestamp: float
    detection_confidence: float
    violation_confidence: float
    evidence_path: str
    annotated_evidence_path: str
    crop_path: Optional[str] = None
    review_status: str
    explanation: str
    created_at: datetime

    class Config:
        from_attributes = True

class ViolationListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: List[ViolationOut]

