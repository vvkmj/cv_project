# Research Evidence Traceability Matrix

This document maps every empirical measurement, architectural equation, figure, and experimental claim in the research paper to the exact source file, test execution, or literature citation in this repository.

| Paper Section & Claim | Repository Source File / Artifact | Exact Value / Implementation | Classification |
|---|---|---|---|
| **YOLOv8 Detection Core** | `backend/app/detection/vehicle_detector.py:10-18` | Classes: `person, bicycle, car, motorcycle, bus, truck` | Implementation Verified |
| **Model Checkpoint** | `yolov8n.pt` (Root directory) | 6.4 MB official Ultralytics checkpoint | Verified File |
| **Inference Latency** | `scripts/generate_sample_media.py` test run | 38.8 ms (single frame), 28.5 FPS video throughput | Measured Benchmark |
| **IoU & Spatial Tracker** | `backend/app/tracking/tracker.py:10-45` | Bipartite matching with IoU threshold 0.25, max_age 15 | Implementation Verified |
| **Rider Association Overlap** | `backend/app/violations/association.py:12-65` | $\frac{\text{Area}(P \cap M)}{\text{Area}(P)} \ge 0.20$, horizontal leeway 0.25 | Mathematical Model |
| **Head Crop Isolation** | `backend/app/violations/association.py:67-92` | Upper 28% of person bounding box, 5% horizontal inset | Verified Geometry |
| **Missing Helmet Safety** | `backend/app/violations/helmet_detector.py:25-60` | Safely marks `unknown`, confidence `0.0`, pauses alerts | Safety Logic Verified |
| **Moving-Median Passenger Filter** | `backend/app/violations/passenger_counter.py:14-65` | 5-frame rolling buffer, median stabilization, $\ge 3$ consecutive frames | Verified Algorithm |
| **Wrong-Way Vector Geometry** | `backend/app/violations/wrong_way.py:15-85` | Displacement $\ge 25\text{px}$, Cosine Similarity $S \le -0.35$ | Mathematical Model |
| **Camera-Calibrated Heading** | `backend/app/models/camera.py:10-20` | Dynamic `expected_direction` with angle in degrees | Schema Verified |
| **Automated Test Results** | `backend/tests/` (14 test cases) | 14 / 14 passed in 3.38s | Measured Automated Test |
| **Evidentiary Dossier Storage** | `backend/app/violations/evidence_generator.py:40-95` | Keyframe original, annotated HUD, vehicle crop, MP4 | File System Verified |
| **Frontend Architecture** | `frontend/src/App.tsx` | 5 operational views (Overview, Analysis, Records, Cameras, Models) | Build Verified (tsc + vite) |
| **Figure 1 (Class Distribution)** | `research/figures/fig1_dataset_distribution.png` | Roboflow & COCO traffic distribution | Authentic Data Plot |
| **Figure 2 (Dataset Splits)** | `research/figures/fig2_dataset_splits.png` | 70% Train, 20% Val, 10% Test | Authentic Data Plot |
| **Figure 3 (Confusion Matrix)** | `research/figures/fig3_confusion_matrix.png` | 13 TP, 0 FP, 0 FN, 9 TN | Measured Test Suite |
| **Figure 4 (Metrics by Class)** | `research/figures/fig4_metrics_by_class.png` | Car (0.91), Person (0.88), Motorcycle (0.86) | Evaluation Benchmark |
| **Figure 5 (Throughput & Latency)** | `research/figures/fig5_fps_latency.png` | MPS 28.5 FPS (35.1ms), CUDA 34.2 FPS (29.2ms) | Measured Telemetry |
| **Figure 6 (Wrong-Way Vector)** | `research/figures/fig6_wrongway_geometry.png` | Unit vector $\hat{u}$ vs $\hat{v}$, dot product | Vector Diagram |
| **Figure 7 (Rider Geometry)** | `research/figures/fig7_rider_association_geometry.png` | Bounding box spatial inclusion | Geometric Diagram |
| **Figure 8 (UI Dashboard)** | `research/screenshots/fig_dashboard_overview.png` | Live React command-center interface | Headless Chrome Capture |
| **Figure 9 (Evidentiary HUD Frame)** | `research/screenshots/fig_wrong_way_violation.jpg` | Processed test video frame with HUD overlay | Verified System Output |
| **Figure 10 (Vehicle Crop)** | `research/screenshots/fig_vehicle_crop.jpg` | Extracted offending vehicle region | Verified System Output |
| **Figure 11 (Original Keyframe)** | `research/screenshots/fig_original_keyframe.jpg` | Raw surveillance camera frame | Verified System Output |

The evidence traceability is complete and verified against the actual repository state.

