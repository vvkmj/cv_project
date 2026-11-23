# AI & Computer Vision Model Architecture Documentation

## 1. System Vision Pipeline Architecture

The **Smart Traffic Violation Detection System** decouples visual perception into independent, testable sub-modules.

```
Input Frame (BGR)
       │
       ▼
[Vehicle & Person Detector (YOLOv8)]
  │ ├── motorcycle, car, bus, truck, person, bicycle
  │ └── Normalized Bounding Boxes [x1, y1, x2, y2]
  ▼
[Spatial & IoU Multi-Object Tracker]
  │ ├── Trajectory History [(t0, x0, y0), (t1, x1, y1)...]
  │ └── Persistent Unique Tracking IDs
  ▼
[Rider-to-Motorcycle Associator]
  │ ├── Overlap Ratio (Intersection / Person Area)
  │ ├── Horizontal Centering & Vertical Posture Validation
  │ └── Rider Head-Crop Extraction (Top 28% of Person BBox)
  ▼
┌──────────────────┬──────────────────┐
│                  │                  │
▼                  ▼                  ▼
[Helmet Classifier] [Passenger Count] [Wrong-Way Vector Analyzer]
- Helmet           - Occupant Count   - Movement Vector (dx, dy)
- No-Helmet        - Buffer Median    - Cosine Similarity vs Camera Vector
- Unknown          - > Limit Trigger  - Min Displacement & Frame Confirm
└──────────────────┴──────────────────┘
       │
       ▼
[Evidence Synthesis & Deduplication Engine]
  ├── Keyframe Original Capture
  ├── Annotated Evidence Frame with HUD
  └── Vehicle Crop & Video Generation
```

---

## 2. Model Components & Specifications

### 2.1 Vehicle & Person Detector
- **Base Architecture:** YOLOv8 (Ultralytics) nano/small (`yolov8n.pt` / `yolov8s.pt`)
- **Backbone:** Modified CSPDarknet53 with C2f (cross-stage partial with 2 convolutions) modules.
- **Neck:** PAN-FPN (Path Aggregation Network with Feature Pyramid Network) for multi-scale feature fusion.
- **Head:** Anchor-free decoupled detection head outputting objectness, class probabilities, and regression offsets.
- **Target Classes:**
  - `0`: Person
  - `1`: Bicycle
  - `2`: Car
  - `3`: Motorcycle
  - `5`: Bus
  - `7`: Truck
- **Default Confidence Threshold:** 0.35 (configurable via API and UI)
- **IoU NMS Threshold:** 0.45

### 2.2 Rider-to-Motorcycle Association
- **Formulation:** A person bounding box $P = [px_1, py_1, px_2, py_2]$ is associated with motorcycle $M = [mx_1, my_1, mx_2, my_2]$ if:
  1. Horizontal Center: $mx_1 - 0.25 \cdot mw \le \frac{px_1+px_2}{2} \le mx_2 + 0.25 \cdot mw$
  2. Vertical Posture: $py_2 \ge my_1$ and $py_1 \le my_1 + 0.85 \cdot mh$
  3. Overlap Ratio: $\frac{\text{Area}(P \cap M)}{\text{Area}(P)} \ge 0.20$
- **Head Region:** Extracted as:
  $$hx_1 = px_1 + 0.05 \cdot pw, \quad hx_2 = px_2 - 0.05 \cdot pw$$
  $$hy_1 = py_1, \quad hy_2 = py_1 + 0.30 \cdot ph$$

### 2.3 Dedicated Helmet Classification
- **Classes:** `helmet`, `no_helmet`, `unknown`.
- **Honest Missing Weights Protocol:**
  - If `ml/weights/helmet_yolov8.pt` is not present, the system marks the detector as `is_available: false`.
  - For any evaluated rider, the status is reported as `unknown`, confidence `0.0`, and explanation: *"Dedicated helmet detection weights not found. Helmet analysis is currently disabled to prevent false accusations."*
  - No synthetic violation records are created.

### 2.4 Multiple-Rider Temporal Stabilizer
- **Buffer:** Rolling window of 5 frames per motorcycle tracking ID.
- **Median Filter:** Uses $\text{median}(\{c_1, c_2, \dots, c_k\})$ to discard transient false detections or occlusions.
- **Verification Rule:** Requires at least 3 consecutive frames with occupant count $> \text{passenger\_limit}$ before generating an alert.

### 2.5 Wrong-Way Directional Vector Analysis
- **Trajectory Window:** Evaluates position history over the last 8 frames:
  $$\vec{v} = (x_k - x_{k-8}, y_k - y_{k-8}), \quad \|\vec{v}\| = \sqrt{\Delta x^2 + \Delta y^2}$$
- **Stationary Vehicle Filter:** If $\|\vec{v}\| < 25.0$ pixels, vector calculation is bypassed (eliminates parked cars, engine vibrations, camera shake).
- **Cosine Similarity:**
  $$\hat{v} = \frac{\vec{v}}{\|\vec{v}\|}, \quad \hat{u} = (\cos \theta_{\text{camera}}, \sin \theta_{\text{camera}})$$
  $$S = \hat{v} \cdot \hat{u} = \cos \phi$$
- **Trigger Condition:** $S \le -0.35$ (angular deviation $> 110.5^\circ$) sustained over $\ge 4$ consecutive frames.
- **Deduplication:** Exactly 1 violation per tracking ID.

---

## 3. Hardware Execution & Acceleration
- **Apple Silicon (Mac M-Series):** Automatically detects `torch.backends.mps.is_available()` and executes on Metal Performance Shaders (MPS).
- **NVIDIA GPU:** Automatically executes on `cuda:0` when CUDA drivers are detected.
- **CPU Fallback:** Multi-threaded CPU execution with PyTorch and OpenCV.

