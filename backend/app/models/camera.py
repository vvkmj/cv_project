from sqlalchemy import Column, String, Integer, JSON, DateTime
from datetime import datetime
from backend.app.database.session import Base

class Camera(Base):
    __tablename__ = "cameras"

    id = Column(String(64), primary_key=True, index=True)
    name = Column(String(128), nullable=False)
    location_label = Column(String(256), nullable=False)
    expected_direction = Column(JSON, nullable=False, default={"angle_deg": 90.0, "dx": 0.0, "dy": 1.0})
    passenger_limit = Column(Integer, default=2, nullable=False)
    enabled_violations = Column(
        JSON,
        nullable=False,
        default={"no_helmet": True, "multiple_riders": True, "wrong_way": True}
    )
    created_at = Column(DateTime, default=datetime.utcnow)

