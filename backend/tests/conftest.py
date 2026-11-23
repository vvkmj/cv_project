import pytest
from backend.app.database.session import Base, engine, SessionLocal
from backend.app.services.camera_service import seed_default_cameras

@pytest.fixture(scope="session", autouse=True)
def initialize_test_database():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    seed_default_cameras(db)
    db.close()
    yield

