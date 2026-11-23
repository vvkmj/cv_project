# REST API Specification & Endpoint Documentation

## 1. Overview

The backend service is powered by **FastAPI** and provides RESTful endpoints under `/api`. All JSON request and response payloads adhere to strict Pydantic schemas.

---

## 2. API Endpoints Reference

### 2.1 Health & Telemetry

#### `GET /api/health`
Returns service availability status and current UTC timestamp.

**Response (200 OK):**
```json
{
  "status": "healthy",
  "service": "Smart Traffic Violation Detection System",
  "version": "1.0.0",
  "timestamp": "2026-11-18T05:30:00.000000"
}
```

#### `GET /api/system/status`
Returns host hardware telemetry and database workload metrics.

**Response (200 OK):**
```json
{
  "system": {
    "cpu_usage_pct": 14.2,
    "memory_usage_pct": 52.8,
    "memory_available_mb": 7820.5,
    "disk_free_gb": 128.45
  },
  "workload": {
    "active_processing_jobs": 0,
    "completed_jobs": 12,
    "total_violations_recorded": 28
  },
  "timestamp": "2026-11-18T05:30:00.000000"
}
```

#### `GET /api/models/status`
Returns active AI model checkpoints, inference device (CPU/MPS/CUDA), and helmet model availability.

**Response (200 OK):**
```json
{
  "vehicle_detector": {
    "name": "Vehicle & Person Detector",
    "model_path": "yolov8n.pt",
    "version": "YOLOv8-yolov8n.pt",
    "is_loaded": true,
    "device": "mps",
    "target_classes": ["person", "bicycle", "car", "motorcycle", "bus", "truck"],
    "confidence_threshold": 0.35
  },
  "helmet_detector": {
    "name": "Dedicated Helmet Detector",
    "weights_path": "ml/weights/helmet_yolov8.pt",
    "version": "Unavailable (No Weights)",
    "is_available": false,
    "device": "cpu",
    "confidence_threshold": 0.65,
    "status_message": "Dedicated helmet detection weights not found. Helmet analysis is currently DISABLED."
  },
  "hardware_device": "mps",
  "pipeline_operational": true
}
```

---

### 2.2 Media Upload & Analysis

#### `POST /api/media/upload`
Uploads an image or video file. Validates format against permitted extensions (`.mp4, .avi, .mov, .jpg, .png`).

**Request:** `multipart/form-data` with `file`.

**Response (200 OK):**
```json
{
  "filename": "f48c3b10ae_traffic.mp4",
  "original_filename": "traffic.mp4",
  "media_type": "video",
  "file_path": "uploads/f48c3b10ae_traffic.mp4",
  "size_bytes": 14205800
}
```

#### `POST /api/analysis/image`
Synchronously processes a static image. Detects vehicles, associates riders, checks multiple occupants, and produces annotated keyframes.

**Request:** `application/x-www-form-urlencoded`
- `filename`: str
- `camera_id`: str (default: "CAM-01")

**Response (200 OK):** `AnalysisJobDetail` schema with full list of detections and violations.

#### `POST /api/analysis/video`
Enqueues asynchronous video analysis as a non-blocking background job.

**Request:** `application/x-www-form-urlencoded`
- `filename`: str
- `camera_id`: str

**Response (202 Accepted):**
```json
{
  "id": "JOB-VID-8F21A04B",
  "filename": "traffic.mp4",
  "media_type": "video",
  "camera_id": "CAM-01",
  "status": "queued",
  "progress": 0.0,
  "total_frames": 1,
  "processed_frames": 0,
  "created_at": "2026-11-18T05:31:00Z"
}
```

#### `GET /api/analysis/{job_id}`
Returns processing telemetry and progress percentage.

#### `GET /api/analysis/{job_id}/results`
Returns completed job results including all tracked detections and generated violation events.

#### `GET /api/analysis/{job_id}/annotated-video`
Streams or downloads the generated annotated MP4 video with visual HUD and tracking overlays.

---

### 2.3 Violation Records

#### `GET /api/violations`
Retrieves a paginated list of violation records with optional query filters:
- `violation_type`: `no_helmet`, `multiple_riders`, `wrong_way`
- `camera_id`: string
- `review_status`: `pending`, `confirmed`, `rejected`
- `min_confidence`: float (0.0 to 1.0)
- `page`: int (default 1)
- `page_size`: int (default 20)

#### `GET /api/violations/{violation_id}`
Fetches full violation details including original evidence, annotated frame, and vehicle crop paths.

#### `PATCH /api/violations/{violation_id}/status`
Updates operator human review decision.

**Request Body:**
```json
{
  "review_status": "confirmed"
}
```

#### `GET /api/violations/export`
Exports violations to file.
- `format`: `csv` or `json`

---

### 2.4 Camera Management

#### `GET /api/cameras`
Lists all configured cameras.

#### `POST /api/cameras`
Creates new camera with expected legal direction vector and violation rules.

**Request Body:**
```json
{
  "id": "CAM-03",
  "name": "East Expressway Toll Plaza",
  "location_label": "Exit 28 Eastbound",
  "expected_direction": {
    "angle_deg": 0.0,
    "dx": 1.0,
    "dy": 0.0,
    "description": "Eastbound flow (Left to Right)"
  },
  "passenger_limit": 2,
  "enabled_violations": {
    "no_helmet": true,
    "multiple_riders": true,
    "wrong_way": true
  }
}
```

#### `PATCH /api/cameras/{camera_id}`
Updates direction angle, passenger limit, or rule toggles.

#### `DELETE /api/cameras/{camera_id}`
Removes camera configuration.

