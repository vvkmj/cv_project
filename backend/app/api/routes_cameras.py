from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.database.session import get_db
from backend.app.schemas.camera import CameraCreate, CameraUpdate, CameraOut
from backend.app.services import camera_service

router = APIRouter(prefix="/cameras", tags=["Camera Management"])

@router.get("", response_model=List[CameraOut])
def list_cameras(db: Session = Depends(get_db)):
    return camera_service.get_cameras(db)

@router.post("", response_model=CameraOut, status_code=status.HTTP_201_CREATED)
def create_camera(camera_in: CameraCreate, db: Session = Depends(get_db)):
    existing = camera_service.get_camera_by_id(db, camera_in.id)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Camera with ID '{camera_in.id}' already exists."
        )
    return camera_service.create_camera(db, camera_in)

@router.get("/{camera_id}", response_model=CameraOut)
def get_camera(camera_id: str, db: Session = Depends(get_db)):
    cam = camera_service.get_camera_by_id(db, camera_id)
    if not cam:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera '{camera_id}' not found."
        )
    return cam

@router.patch("/{camera_id}", response_model=CameraOut)
def update_camera(camera_id: str, camera_in: CameraUpdate, db: Session = Depends(get_db)):
    cam = camera_service.update_camera(db, camera_id, camera_in)
    if not cam:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera '{camera_id}' not found."
        )
    return cam

@router.delete("/{camera_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_camera(camera_id: str, db: Session = Depends(get_db)):
    success = camera_service.delete_camera(db, camera_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera '{camera_id}' not found."
        )
    return None

