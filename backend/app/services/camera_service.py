from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from backend.app.models.camera import Camera
from backend.app.schemas.camera import CameraCreate, CameraUpdate

DEFAULT_CAMERAS = [
    {
        "id": "CAM-01",
        "name": "North Junction Highway Approach",
        "location_label": "Interstate 95 - Milepost 14 Northbound",
        "expected_direction": {
            "angle_deg": 90.0,
            "dx": 0.0,
            "dy": 1.0,
            "description": "Southbound flow (Top to Bottom)"
        },
        "passenger_limit": 2,
        "enabled_violations": {
            "no_helmet": True,
            "multiple_riders": True,
            "wrong_way": True
        }
    },
    {
        "id": "CAM-02",
        "name": "Downtown Urban Corridor",
        "location_label": "Broadway Ave & 4th Street",
        "expected_direction": {
            "angle_deg": 180.0,
            "dx": -1.0,
            "dy": 0.0,
            "description": "Westbound flow (Right to Left)"
        },
        "passenger_limit": 2,
        "enabled_violations": {
            "no_helmet": True,
            "multiple_riders": True,
            "wrong_way": True
        }
    }
]

def seed_default_cameras(db: Session):
    for cam_data in DEFAULT_CAMERAS:
        existing = db.query(Camera).filter(Camera.id == cam_data["id"]).first()
        if not existing:
            cam = Camera(**cam_data)
            db.add(cam)
    db.commit()

def get_cameras(db: Session) -> List[Camera]:
    return db.query(Camera).all()

def get_camera_by_id(db: Session, camera_id: str) -> Optional[Camera]:
    return db.query(Camera).filter(Camera.id == camera_id).first()

def create_camera(db: Session, camera_in: CameraCreate) -> Camera:
    cam = Camera(**camera_in.model_dump())
    db.add(cam)
    db.commit()
    db.refresh(cam)
    return cam

def update_camera(db: Session, camera_id: str, camera_in: CameraUpdate) -> Optional[Camera]:
    cam = get_camera_by_id(db, camera_id)
    if not cam:
        return None
    
    update_data = camera_in.model_dump(exclude_unset=True)
    for field, val in update_data.items():
        setattr(cam, field, val)
        
    db.commit()
    db.refresh(cam)
    return cam

def delete_camera(db: Session, camera_id: str) -> bool:
    cam = get_camera_by_id(db, camera_id)
    if not cam:
        return False
    db.delete(cam)
    db.commit()
    return True

