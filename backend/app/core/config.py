import os
from pydantic import BaseModel

class Settings(BaseModel):
    PROJECT_NAME: str = "Smart Traffic Violation Detection System"
    VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api"
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./traffic_system.db")
    
    # Storage Paths
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "uploads")
    EVIDENCE_DIR: str = os.getenv("EVIDENCE_DIR", "evidence")
    SAMPLE_MEDIA_DIR: str = os.getenv("SAMPLE_MEDIA_DIR", "sample_media")
    
    # Model Paths
    VEHICLE_MODEL_PATH: str = os.getenv("VEHICLE_MODEL_PATH", "yolov8n.pt")
    HELMET_MODEL_PATH: str = os.getenv("HELMET_MODEL_PATH", "ml/weights/helmet_yolov8.pt")
    
    # Default Thresholds
    DETECTION_CONFIDENCE: float = float(os.getenv("DETECTION_CONFIDENCE", "0.35"))
    HELMET_CONFIDENCE: float = float(os.getenv("HELMET_CONFIDENCE", "0.50"))
    DEFAULT_PASSENGER_LIMIT: int = int(os.getenv("DEFAULT_PASSENGER_LIMIT", "2"))

settings = Settings()

# Ensure directories exist
for folder in [
    settings.UPLOAD_DIR,
    settings.EVIDENCE_DIR,
    f"{settings.EVIDENCE_DIR}/original",
    f"{settings.EVIDENCE_DIR}/annotated",
    f"{settings.EVIDENCE_DIR}/crops",
    f"{settings.EVIDENCE_DIR}/videos",
    settings.SAMPLE_MEDIA_DIR,
    "ml/weights"
]:
    os.makedirs(folder, exist_ok=True)
