from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey, Text
from datetime import datetime
from backend.app.database.session import Base

class Violation(Base):
    __tablename__ = "violations"

    id = Column(String(64), primary_key=True, index=True)
    job_id = Column(String(64), ForeignKey("analysis_jobs.id"), nullable=False, index=True)
    camera_id = Column(String(64), ForeignKey("cameras.id"), nullable=False, index=True)
    tracking_id = Column(Integer, nullable=True)
    violation_type = Column(String(64), nullable=False, index=True) # no_helmet, multiple_riders, wrong_way
    timestamp = Column(Float, default=0.0)
    detection_confidence = Column(Float, default=0.0)
    violation_confidence = Column(Float, default=0.0)
    evidence_path = Column(String(512), nullable=False)
    annotated_evidence_path = Column(String(512), nullable=False)
    crop_path = Column(String(512), nullable=True)
    review_status = Column(String(32), default="pending", nullable=False, index=True) # pending, confirmed, rejected
    explanation = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

