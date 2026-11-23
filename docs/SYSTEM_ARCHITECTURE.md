# System Architecture & Technical Specification

## 1. Architectural Philosophy

The **Smart Traffic Violation Detection System** is designed around the principles of **modular decoupling**, **fail-safe execution**, and **human-in-the-loop decision support**.

The system avoids monolitihic computer vision scripts by partitioning responsibilities into clean, testable sub-layers:

```
[Camera Streams / Video Uploads]
             │
             ▼
┌──────────────────────────────────────────────┐
│ FastAPI Backend Service Layer                │
│ ├── Media Ingestion & Validation             │
│ ├── Background Task Worker                   │
│ ├── Database ORM (SQLAlchemy + SQLite)       │
│ └── Static Media & Evidence File Server      │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ Computer Vision & Analytics Pipeline         │
│ ├── Frame Extraction (OpenCV)                │
│ ├── Vehicle & Person Detection (YOLOv8)      │
│ ├── Multi-Object Tracking (Spatial / IoU)    │
│ ├── Rider-to-Motorcycle Spatial Associator   │
│ ├── Head-Crop Extraction & Helmet Classifier │
│ ├── Temporal Multi-Rider Passenger Counter   │
│ └── Directional Wrong-Way Vector Analyzer    │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ Storage & Evidentiary Assets                 │
│ ├── SQLite DB (Jobs, Cameras, Violations)    │
│ ├── Evidence Keyframes (Original & Annotated)│
│ ├── Vehicle Crops                            │
│ └── Annotated Video MP4 Streams              │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ React TypeScript Dashboard (Tailwind CSS)    │
│ ├── Overview KPI Command Center              │
│ ├── Media Analysis & Live Telemetry          │
│ ├── Violation Records & Audit Modal          │
│ ├── Camera & Direction Compass Configurator  │
│ └── AI Model Health & Limitations Matrix     │
└──────────────────────────────────────────────┘
```

---

## 2. Database Schema Design (Entity-Relationship)

```
┌────────────────────────┐         ┌────────────────────────┐
│        cameras         │         │     analysis_jobs      │
├────────────────────────┤         ├────────────────────────┤
│ id (PK, String)        │◄───┐    │ id (PK, String)        │◄───┐
│ name (String)          │    │    │ filename (String)      │    │
│ location_label (String)│    │    │ media_type (String)    │    │
│ expected_direction(JSN)│    │    │ camera_id (FK)         │    │
│ passenger_limit (Int)  │    │    │ status (String)        │    │
│ enabled_violations(JSN)│    │    │ progress (Float)       │    │
│ created_at (DateTime)  │    │    │ total_frames (Int)     │    │
└────────────────────────┘    │    │ processed_frames (Int) │    │
                              │    │ summary (JSON)         │    │
                              │    │ annotated_path (String)│    │
                              │    │ created_at (DateTime)  │    │
                              │    └────────────────────────┘    │
                              │                                  │
                              │    ┌────────────────────────┐    │
                              │    │       violations       │    │
                              │    ├────────────────────────┤    │
                              ├───┼│ id (PK, String)        │    │
                              │    │ job_id (FK)            ├────┘
                              └───┼│ camera_id (FK)         │
                                   │ tracking_id (Int)      │
                                   │ violation_type (String)│
                                   │ timestamp (Float)      │
                                   │ detection_conf (Float) │
                                   │ violation_conf (Float) │
                                   │ evidence_path (String) │
                                   │ annot_evidence_path(S) │
                                   │ crop_path (String)     │
                                   │ review_status (String) │
                                   │ explanation (Text)     │
                                   │ created_at (DateTime)  │
                                   └────────────────────────┘
```

---

## 3. Computer Vision Algorithmic Pipeline

### 3.1 Object Detection
- **Model:** YOLOv8 nano/small (`yolov8n.pt`)
- **Classes Filtered:** `person`, `motorcycle`, `car`, `bus`, `truck`, `bicycle`.
- **Bounding Box Format:** $[x_1, y_1, x_2, y_2]$ in pixel coordinates.

### 3.2 Spatial & IoU Multi-Object Tracking
- Maintains persistent `Track` structures across video frames.
- Cost matrix computed via Intersection over Union ($\text{IoU}$).
- Trajectory smoothing and velocity computation over moving historical window.

### 3.3 Rider-to-Motorcycle Geometric Association
- A person is associated with a motorcycle when:
  1. $\text{Center}_x(\text{Person})$ is within $[mx_1 - 0.25 \cdot mw, mx_2 + 0.25 \cdot mw]$
  2. $\text{Vertical overlap}$ sits on or inside the upper half of the motorcycle body.
  3. Spatial intersection over person area $\ge 0.20$.
- Head region is extracted dynamically: top 28% of person bounding box.

### 3.4 Temporal Multiple-Rider Verification
- Motorcycle occupant counts are collected into a 5-frame rolling buffer.
- Median filter computes $\text{median}(\text{buffer})$ to eliminate single-frame noise or occlusions.
- Violation triggered only when count consistently exceeds `passenger_limit` over $\ge 3$ consecutive frames.

### 3.5 Camera-Specific Direction & Wrong-Way Analysis
- Computes vehicle trajectory displacement vector $\vec{v} = (\Delta x, \Delta y)$ over 8 frames.
- Minimum displacement threshold: $\|\vec{v}\| \ge 25.0$ pixels (filters out stationary vehicles, vibrations).
- Computes Cosine Similarity against camera legal heading vector $\hat{u}$:
  $$S = \hat{v} \cdot \hat{u} = \cos \phi$$
- If $S \le -0.35$ (opposing flow) sustained over $\ge 3$ consecutive observations, triggers violation.
- Deduplication prevents repetitive alerts for the same continuous track.

