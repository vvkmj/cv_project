# Smart Traffic Violation Detection System
### Autonomous Multi-Rule Computer Vision & Decision-Support Platform

[![Python](https://img.shields.io/badge/Python-3.13%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.2-61DAFB.svg)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.2-blue.svg)](https://www.typescriptlang.org/)
[![Ultralytics](https://img.shields.io/badge/YOLO-v8-00FFFF.svg)](https://ultralytics.com/)
[![Tests](https://img.shields.io/badge/Tests-14%2F14%20Passing-brightgreen.svg)]()

---

## 1. Project Introduction
The **Smart Traffic Violation Detection System** is an end-to-end, real-time Computer Vision and decision-support application engineered to identify multiple traffic infractions from traffic-camera images and video feeds.

Rather than drawing generic bounding boxes or issuing automated fines without verification, the platform tracks vehicular dynamics, enforces geometric rider-vehicle association, performs temporal multi-rider counting, evaluates camera-specific wrong-way driving vectors, and produces tamper-evident dossiers (original keyframes, annotated frames, close-up crops, and annotated MP4 video clips) for human operator review.

---

## 2. Problem Statement
Urban roadway safety enforcement relies heavily on manual surveillance or static speed traps. Manual monitoring cannot scale to millions of vehicles, while conventional ANPR cameras are blind to dynamic violations like:
- **Riding without protective headgear (No Helmet)**
- **Overcrowded motorcycles exceeding legal occupancy (Triple-Riding / Multiple Riders)**
- **Driving against authorized traffic flow (Wrong-Way Driving)**

Existing naive AI prototypes suffer from extreme false positives by mistaking nearby pedestrians for riders, inferring helmet status from full bodies, or assuming hardcoded screen directions.

---

## 3. Objectives
1. **Multi-Class Vehicle & Person Detection:** Detect motorcycles, cars, buses, trucks, pedestrians, and bicycles using YOLOv8.
2. **Rider-to-Motorcycle Association:** Geometrically pair riders with their motorcycles while rejecting roadside pedestrians.
3. **Dedicated Helmet Classification:** Isolate rider head regions and classify headgear compliance with scientific integrity (honestly pausing alerts when custom weights are unavailable).
4. **Temporal Passenger Counting:** Enforce configurable passenger limits (default: 2) with moving-median stabilization to prevent flicker.
5. **Directional Wrong-Way Analysis:** Track vehicles across frames and compute trajectory cosine similarity against camera-calibrated legal vectors.
6. **Evidentiary Dossier Generation:** Persist original keyframes, annotated frames with HUD banners, vehicle crops, and processed video clips.
7. **Interactive Command-Center Dashboard:** Modern web interface with real-time KPI metrics, video progress tracking, audit ledger, and 360° compass heading calibration.

---

## 4. Existing System Limitations vs. Proposed Solution

| Feature | Conventional / Naive Systems | Smart Traffic Violation Detection System |
|---|---|---|
| **Helmet Detection** | Inferred from full-body box (causes false positives from jackets/hats) | **Two-stage**: Isolates upper 28% head region for dedicated classification. |
| **Missing Model Handling** | Fabricates fake predictions or crashes silently | **Honest Protocol**: Safely pauses alerts, marks status as `Unknown`, and explains how to install weights. |
| **Passenger Counting** | Single-frame count (vulnerable to occlusion) | **Temporal moving-median filter** across 5-frame rolling buffers. |
| **Wrong-Way Driving** | Hardcoded screen directions (assumes left/right) | **Camera-specific heading vectors** calibrated via interactive compass. |
| **Operator Workflow** | Automated black-box ticketing | **Human-in-the-loop audit ledger** with Confirm/Reject actions and CSV/JSON export. |

---

## 5. System Architecture

```
[Traffic Camera Feeds / Uploads]
             │
             ▼
┌──────────────────────────────────────────────┐
│ FastAPI Backend Application                  │
│ ├── Async Media Ingestion & Validation       │
│ ├── Background Task Worker                   │
│ ├── Database Transactions (SQLite + ORM)     │
│ └── Static Asset & Evidence File Server      │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ Computer Vision & Analytics Engine           │
│ ├── OpenCV Frame Extraction                  │
│ ├── YOLOv8 Vehicle & Person Detection        │
│ ├── Multi-Object Tracking (Spatial / IoU)    │
│ ├── Rider-to-Motorcycle Spatial Associator   │
│ ├── Dedicated Head-Crop Helmet Classifier    │
│ ├── Moving-Median Passenger Counter          │
│ └── Wrong-Way Trajectory Vector Analyzer     │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ Evidentiary Synthesis Layer                  │
│ ├── Original Keyframe Capture                │
│ ├── Annotated Evidence Keyframe with HUD     │
│ ├── Offending Vehicle & Rider Crops          │
│ └── Annotated Video MP4 Streams              │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ React TypeScript Dashboard (Tailwind CSS)    │
│ ├── Overview KPI Command Center              │
│ ├── Upload & Live Progress Analysis Player   │
│ ├── Searchable Violation Audit Ledger        │
│ ├── Camera & 360° Compass Configurator       │
│ └── AI Model Health & Telemetry Matrix       │
└──────────────────────────────────────────────┘
```

---

## 6. Technology Stack
- **AI & Computer Vision:** Python 3.13, PyTorch 2.15, Ultralytics YOLOv8, OpenCV 5.1, NumPy, SciPy, Pandas.
- **Backend Framework:** FastAPI, Uvicorn, Pydantic v2, SQLAlchemy 2.0, SQLite.
- **Frontend Dashboard:** React 18, TypeScript 5, Vite 5, Tailwind CSS 3, Lucide React, Recharts.
- **QA & Testing:** Pytest 9, Pytest-Asyncio, HTTPX TestClient.
- **DevOps & Packaging:** Docker, Docker Compose, Virtualenv.

---

## 7. AI & Computer Vision Methodology

### 7.1 Rider-to-Motorcycle Geometric Association
A candidate person bounding box $P = [px_1, py_1, px_2, py_2]$ is associated with motorcycle $M = [mx_1, my_1, mx_2, my_2]$ when:
1. **Horizontal Centering:** $mx_1 - 0.25 \cdot mw \le \frac{px_1 + px_2}{2} \le mx_2 + 0.25 \cdot mw$
2. **Vertical Posture:** $py_2 \ge my_1$ and $py_1 \le my_1 + 0.85 \cdot mh$
3. **Spatial Overlap:** $\frac{\text{Area}(P \cap M)}{\text{Area}(P)} \ge 0.20$

### 7.2 Dedicated Helmet Classification
The head region $H$ is extracted as the upper 28% of the rider's bounding box:
$$hx_1 = px_1 + 0.05 \cdot pw, \quad hx_2 = px_2 - 0.05 \cdot pw, \quad hy_1 = py_1, \quad hy_2 = py_1 + 0.28 \cdot ph$$
If custom weights are loaded, the classifier evaluates `helmet`, `no_helmet`, or `unknown`. If weights are missing, the system outputs `unknown` without raising alerts.

### 7.3 Temporal Multiple-Rider Stabilization
Occupant counts over a 5-frame rolling buffer are evaluated with a median filter:
$$C^*_\tau = \text{median}(\{c_{t-4}, \dots, c_t\})$$
An alert is generated only if $C^*_\tau > \text{passenger\_limit}$ sustained over $\ge 3$ consecutive frames.

### 7.4 Directional Wrong-Way Trajectory Analysis
Displacement vector $\vec{v} = (\Delta x, \Delta y)$ is computed over 8 frames. If $\|\vec{v}\| \ge 25\text{px}$, Cosine Similarity against camera heading $\hat{u}$ is calculated:
$$S = \hat{v} \cdot \hat{u} = \cos \phi$$
If $S \le -0.35$ sustained over $\ge 3$ consecutive frames, a wrong-way violation is verified. Exactly one alert per vehicle track is generated (deduplication).

---

## 8. Dataset Strategy & Model Training

### 8.1 Public Datasets
- **Roboflow Motorcycle Helmet Detection:** 5,120 annotated surveillance images under Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Kaggle Motorcycle Helmet Dataset:** Open Data Commons Public Domain (CC0).
- **UA-DETRAC Benchmark:** 140,000 traffic surveillance video frames for multi-vehicle tracking validation.

### 8.2 Reproducible Dataset Preparation & Training
```bash
# Prepare dataset splits (70% train, 20% val, 10% test)
./venv/bin/python3 ml/datasets/dataset_prep.py

# Execute transfer learning for custom helmet detection
./venv/bin/python3 ml/training/train_helmet.py --epochs 30 --batch 16
```

---

## 9. Installation & Execution Guide

### 9.1 Prerequisites
- macOS (Apple Silicon or Intel), Linux, or Windows (WSL2)
- Python 3.10 to 3.13
- Node.js v18+ and npm v9+
- Git

### 9.2 Step-by-Step Installation

```bash
# 1. Clone the repository and navigate into the workspace
git clone <repo-url>
cd cv_project

# 2. Set up Python virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install backend dependencies
./venv/bin/pip install -r backend/requirements.txt

# 4. Install frontend npm dependencies
cd frontend
npm install
cd ..

# 5. Generate sample traffic media assets
./venv/bin/python3 scripts/generate_sample_media.py
```

### 9.3 Running the Application

Open two terminal sessions:

**Terminal 1 — Backend API Service:**
```bash
./venv/bin/uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```
- API Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Healthcheck: [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

**Terminal 2 — Frontend Command-Center Dashboard:**
```bash
cd frontend
npm run dev
```
- Open browser at [http://localhost:3000](http://localhost:3000)

---

## 10. Automated Testing & Verification

Run the comprehensive pytest suite covering computer vision algorithms, tracking, geometry, REST APIs, and end-to-end workflows:

```bash
./venv/bin/pytest -v
```

**Test Execution Results:**
```
============================== 14 passed in 3.38s ==============================
backend/tests/test_e2e_workflow.py::test_full_video_processing_end_to_end PASSED
backend/tests/test_integration_api.py::test_health_endpoints PASSED
backend/tests/test_integration_api.py::test_camera_crud_lifecycle PASSED
backend/tests/test_integration_api.py::test_media_upload_and_validation PASSED
backend/tests/test_integration_api.py::test_image_analysis_pipeline PASSED
backend/tests/test_integration_api.py::test_violations_listing_filtering_and_export PASSED
backend/tests/test_integration_api.py::test_violation_status_update PASSED
backend/tests/test_unit_cv.py::test_iou_calculation PASSED
backend/tests/test_unit_cv.py::test_multiobject_tracker PASSED
backend/tests/test_unit_cv.py::test_rider_associator PASSED
backend/tests/test_unit_cv.py::test_head_region_extraction PASSED
backend/tests/test_unit_cv.py::test_helmet_detector_missing_weights_handling PASSED
backend/tests/test_unit_cv.py::test_passenger_counter PASSED
backend/tests/test_unit_cv.py::test_wrong_way_detector PASSED
```

---

## 11. Folder Structure

```
cv_project/
│
├── backend/
│   ├── app/
│   │   ├── api/                  # RESTful API route endpoints
│   │   ├── core/                 # App configuration & settings
│   │   ├── database/             # SQLite connection & session
│   │   ├── models/               # SQLAlchemy ORM models (Camera, Job, Violation)
│   │   ├── schemas/              # Pydantic validation schemas
│   │   ├── services/             # Background job runner & camera services
│   │   ├── detection/            # YOLOv8 vehicle detector & analysis pipeline
│   │   ├── tracking/             # Spatial & IoU multi-object tracker
│   │   ├── violations/           # Association, helmet, passenger & wrong-way analyzers
│   │   └── main.py               # FastAPI entrypoint & middleware
│   ├── tests/                    # Unit, integration & e2e test suites
│   ├── Dockerfile
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── components/           # Navbar, EvidenceModal, UI elements
│   │   ├── pages/                # Overview, Analysis, Violations, Cameras, ModelStatus
│   │   ├── services/             # API client interface
│   │   ├── types/                # TypeScript interfaces
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── package.json
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   └── Dockerfile
│
├── ml/
│   ├── configs/                  # Training YAML configurations
│   ├── datasets/                 # Ingestion & dataset prep scripts
│   ├── evaluation/               # Model benchmark & evaluation metrics
│   ├── training/                 # Transfer learning training scripts
│   └── weights/                  # Directory for custom model checkpoints
│
├── docs/
│   ├── PROJECT_REPORT.md         # Comprehensive academic report
│   ├── PRESENTATION_OUTLINE.md   # 12-slide viva deck outline
│   ├── DEMO_SCRIPT.md            # Step-by-step viva presentation script
│   ├── SYSTEM_ARCHITECTURE.md    # Architecture blueprints & ER diagrams
│   ├── API_DOCUMENTATION.md      # REST API reference
│   ├── DATASET_DOCUMENTATION.md  # Public datasets audit & licensing
│   ├── MODEL_DOCUMENTATION.md    # Neural network & tracking formulas
│   └── TESTING_REPORT.md         # Formal QA testing matrix
│
├── sample_media/                 # Generated realistic traffic test assets
├── evidence/                     # Generated evidentiary keyframes, crops, and MP4 videos
├── scripts/                      # Utility and sample generation scripts
├── docker-compose.yml
├── pytest.ini
├── .env.example
├── .gitignore
└── README.md
```

---

## 12. Ethical & Regulatory Compliance
- **Human-in-the-Loop:** Designed strictly as an operator decision-support platform. The software never automatically issues monetary penalties.
- **Privacy by Design:** No facial recognition is implemented. License plate reading (ANPR) is disabled by default to comply with regional privacy regulations.
- **Audit Ledger:** Every violation records model version, confidence scores, and mathematical reasoning for compliance verification.

