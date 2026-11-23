from sqlalchemy import Column, String, Integer, Float, JSON, DateTime, ForeignKey
from datetime import datetime
from backend.app.database.session import Base

class AnalysisJob(Base):
    __tablename__ = "analysis_jobs"

    id = Column(String(64), primary_key=True, index=True)
    filename = Column(String(256), nullable=False)
    media_type = Column(String(32), nullable=False) # 'image' or 'video'
    camera_id = Column(String(64), ForeignKey("cameras.id"), nullable=False, default="CAM-01")
    status = Column(String(32), nullable=False, default="queued") # queued, processing, completed, failed
    progress = Column(Float, default=0.0)
    total_frames = Column(Integer, default=1)
    processed_frames = Column(Integer, default=0)
    summary = Column(JSON, nullable=True)
    annotated_path = Column(String(512), nullable=True)
    error_message = Column(String(1024), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

