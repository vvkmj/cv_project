# Agent Communication Interfaces & Data Contracts

## 1. Overview

This document specifies the shared data schemas, function signatures, and API contracts for the **Smart Traffic Violation Detection System**. All agents (AI/CV, Backend, Frontend, Dataset, QA, Documentation) MUST strictly adhere to these contracts to guarantee seamless integration.

---

## 2. Shared Data Contracts

### 2.1 Detection Result Contract (`DetectionResult`)
Represents an individual object detected in a media frame (image or video frame).

```json
{
  "frame_number": 0,
  "timestamp": 0.0,
  "tracking_id": 1,
  "object_class": "motorcycle",
  "confidence": 0.89,
  "bounding_box": {
    "x1": 120.5,
    "y1": 240.0,
    "x2": 310.0,
    "y2": 520.5
  },
  "model_version": "yolov8n-coco-8.1"
}
```

Python Schema Definition (`backend/app/schemas/detection.py`):
- `frame_number: int`
- `timestamp: float` (seconds from video start or 0.0 for static image)
- `tracking_id: Optional[int]`
- `object_class: str` (e.g., `motorcycle`, `person`, `car`, `bus`, `truck`, `bicycle`)
- `confidence: float` (0.0 - 1.0)
- `bounding_box: BoundingBox` (`x1: float, y1: float, x2: float, y2: float`)
- `model_version: str`
- `associated_rider_ids: List[int]` (optional, for motorcycles)
- `helmet_status: Optional[str]` (for riders: `"helmet"`, `"no_helmet"`, `"unknown"`)

---

### 2.2 Violation Result Contract (`ViolationResult`)
Represents an identified, verified traffic violation event with associated evidentiary artifact paths.

```json
{
  "violation_id": "VIOL-20261117-0001",
  "job_id": "JOB-20261117-0001",
  "camera_id": "CAM-01",
  "tracking_id": 1,
  "violation_type": "no_helmet",
  "timestamp": 12.4,
  "detection_confidence": 0.91,
  "violation_confidence": 0.88,
  "evidence_path": "evidence/original/VIOL-20261117-0001.jpg",
  "annotated_evidence_path": "evidence/annotated/VIOL-20261117-0001.jpg",
  "crop_path": "evidence/crops/VIOL-20261117-0001_crop.jpg",
  "explanation": "Motorcycle rider (Track #1) detected without helmet with confidence 0.88",
  "review_status": "pending",
  "created_at": "2026-11-17T23:30:00Z"
}
```

Permitted Values:
- `violation_type`: `"no_helmet"`, `"multiple_riders"`, `"wrong_way"`
- `review_status`: `"pending"`, `"confirmed"`, `"rejected"`

---

### 2.3 Processing Job Contract (`JobResult`)
Represents asynchronous analysis job execution status and telemetry.

```json
{
  "job_id": "JOB-20261117-0001",
  "filename": "traffic_test_01.mp4",
  "media_type": "video",
  "status": "processing",
  "progress": 45.2,
  "total_frames": 300,
  "processed_frames": 136,
  "created_at": "2026-11-17T23:25:00Z",
  "completed_at": null,
  "error_message": null,
  "summary": {
    "total_vehicles_detected": 14,
    "violations_count": 2,
    "no_helmet_count": 1,
    "multiple_riders_count": 1,
    "wrong_way_count": 0
  }
}
```

Permitted Values for `status`:
- `"queued"`
- `"processing"`
- `"completed"`
- `"failed"`

---

### 2.4 Camera Configuration Contract (`CameraConfig`)
Defines the spatial and violation rules governing each monitoring stream/camera.

```json
{
  "id": "CAM-01",
  "name": "North Junction Approach",
  "location_label": "Main Street & 5th Avenue",
  "expected_direction": {
    "angle_deg": 90.0,
    "dx": 0.0,
    "dy": 1.0,
    "description": "Southbound flow (Top to Bottom)"
  },
  "passenger_limit": 2,
  "enabled_violations": {
    "no_helmet": true,
    "multiple_riders": true,
    "wrong_way": true
  },
  "created_at": "2026-11-17T23:20:00Z"
}
```

---

## 3. Computer Vision Module Interface (`backend/app/detection/pipeline.py`)

The Computer Vision engine must expose the following interface to the Backend:

```python
class TrafficAnalysisPipeline:
    def __init__(self, config: PipelineConfig):
        ...

    def analyze_image(
        self,
        image_path: str,
        camera_config: CameraConfig,
        job_id: str
    ) -> AnalysisResult:
        """Processes a static frame; returns detections and verified violations."""
        ...

    def analyze_video(
        self,
        video_path: str,
        camera_config: CameraConfig,
        job_id: str,
        progress_callback: Optional[Callable[[float, int, int], None]] = None
    ) -> AnalysisResult:
        """
        Processes video stream frame-by-frame:
        1. Tracks objects across frames.
        2. Applies rider-motorcycle spatial association.
        3. Analyzes helmet status (or marks Unknown if weights absent).
        4. Verifies multiple riders using temporal consistency (> threshold).
        5. Computes trajectory vectors and verifies wrong-way violations.
        6. Generates evidence crops, annotated keyframes, and output video.
        """
        ...

    def get_model_status(self) -> Dict[str, Any]:
        """Returns loaded model weights, device (CPU/MPS/CUDA), and helmet status."""
        ...
```

---

## 4. REST API Contract Overview

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Healthcheck |
| `GET` | `/api/system/status` | System telemetry & resources |
| `GET` | `/api/models/status` | Active weights and capabilities |
| `POST` | `/api/media/upload` | Upload video/image file |
| `POST` | `/api/analysis/image` | Run analysis on image synchronously |
| `POST` | `/api/analysis/video` | Enqueue background video analysis |
| `GET` | `/api/analysis/{job_id}` | Poll analysis job status and progress |
| `GET` | `/api/analysis/{job_id}/results` | Fetch complete detections & violations |
| `GET` | `/api/analysis/{job_id}/annotated-video` | Stream or download annotated MP4 |
| `GET` | `/api/violations` | Filterable list of violation records |
| `GET` | `/api/violations/{violation_id}` | Retrieve specific violation & evidence |
| `PATCH` | `/api/violations/{violation_id}/status` | Update review status (confirmed/rejected) |
| `GET` | `/api/violations/export` | Export CSV or JSON report |
| `GET` | `/api/cameras` | List configured cameras |
| `POST` | `/api/cameras` | Create new camera configuration |
| `PATCH` | `/api/cameras/{camera_id}` | Update direction/threshold parameters |
| `DELETE` | `/api/cameras/{camera_id}` | Remove camera |

