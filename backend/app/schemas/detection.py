from pydantic import BaseModel
from typing import Optional, Dict, Any

class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

class DetectionOut(BaseModel):
    id: Optional[int] = None
    frame_number: int
    timestamp: float
    tracking_id: Optional[int] = None
    object_class: str
    confidence: float
    bounding_box: Dict[str, float]

    class Config:
        from_attributes = True

