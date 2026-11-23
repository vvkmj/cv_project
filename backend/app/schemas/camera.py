from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime

class ExpectedDirectionSchema(BaseModel):
    angle_deg: Optional[float] = 90.0
    dx: Optional[float] = 0.0
    dy: Optional[float] = 1.0
    description: Optional[str] = "Southbound flow"

class CameraBase(BaseModel):
    id: str = Field(..., example="CAM-01")
    name: str = Field(..., example="North Junction Main Gate")
    location_label: str = Field(..., example="Highway 101 KM 42")
    expected_direction: Dict[str, Any] = Field(default_factory=lambda: {"angle_deg": 90.0, "dx": 0.0, "dy": 1.0})
    passenger_limit: int = Field(default=2, ge=1, le=5)
    enabled_violations: Dict[str, bool] = Field(
        default_factory=lambda: {"no_helmet": True, "multiple_riders": True, "wrong_way": True}
    )

class CameraCreate(CameraBase):
    pass

class CameraUpdate(BaseModel):
    name: Optional[str] = None
    location_label: Optional[str] = None
    expected_direction: Optional[Dict[str, Any]] = None
    passenger_limit: Optional[int] = None
    enabled_violations: Optional[Dict[str, bool]] = None

class CameraOut(CameraBase):
    created_at: datetime

    class Config:
        from_attributes = True

