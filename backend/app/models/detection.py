from sqlalchemy import Column, Integer, String, Float, JSON, ForeignKey
from backend.app.database.session import Base

class Detection(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(64), ForeignKey("analysis_jobs.id"), nullable=False, index=True)
    tracking_id = Column(Integer, nullable=True, index=True)
    frame_number = Column(Integer, default=0, index=True)
    object_class = Column(String(64), nullable=False)
    confidence = Column(Float, nullable=False)
    bounding_box = Column(JSON, nullable=False)
    timestamp = Column(Float, default=0.0)

