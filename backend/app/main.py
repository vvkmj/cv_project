import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.core.config import settings
from backend.app.database.session import engine, Base, SessionLocal
from backend.app.services.camera_service import seed_default_cameras
from backend.app.api.routes_health import router as health_router
from backend.app.api.routes_cameras import router as cameras_router
from backend.app.api.routes_analysis import router as analysis_router
from backend.app.api.routes_violations import router as violations_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("TrafficSystem")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="End-to-End Smart Traffic Violation Detection System API"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static asset directories for evidence and uploads
for dir_path in [settings.EVIDENCE_DIR, settings.UPLOAD_DIR, settings.SAMPLE_MEDIA_DIR]:
    os.makedirs(dir_path, exist_ok=True)

app.mount("/evidence", StaticFiles(directory=settings.EVIDENCE_DIR), name="evidence")
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")
app.mount("/sample_media", StaticFiles(directory=settings.SAMPLE_MEDIA_DIR), name="sample_media")

# Include API Routers under /api
app.include_router(health_router, prefix=settings.API_V1_PREFIX)
app.include_router(cameras_router, prefix=settings.API_V1_PREFIX)
app.include_router(analysis_router, prefix=settings.API_V1_PREFIX)
app.include_router(violations_router, prefix=settings.API_V1_PREFIX)

@app.on_event("startup")
def startup_event():
    logger.info("Initializing database schema...")
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        seed_default_cameras(db)
        logger.info("Database initialized and default cameras verified.")
    finally:
        db.close()

@app.get("/")
def root():
    return {
        "message": f"Welcome to {settings.PROJECT_NAME}",
        "docs_url": "/docs",
        "api_v1": settings.API_V1_PREFIX
    }

