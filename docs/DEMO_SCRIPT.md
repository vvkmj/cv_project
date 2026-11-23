# Faculty Viva & Demonstration Script
## Smart Traffic Violation Detection System: Step-by-Step Viva Presentation Guide

**Target Duration:** 10–12 Minutes &bull; **Purpose:** Demonstration of Working Features for Faculty Evaluation & Defense

---

### Step 1: Pre-Demo Setup Verification (1 Minute)
Before the evaluators arrive, ensure both backend and frontend servers are operational.

1. **Terminal 1 (Backend Service):**
   ```bash
   cd /Users/vivekmaanju/Downloads/cv_project
   ./venv/bin/uvicorn backend.app.main:app --reload --port 8000
   ```
   *Expected output:* `INFO: Application startup complete. Uvicorn running on http://127.0.0.1:8000`

2. **Terminal 2 (Frontend Dashboard):**
   ```bash
   cd /Users/vivekmaanju/Downloads/cv_project/frontend
   npm run dev
   ```
   *Expected output:* `Local: http://localhost:3000/`

3. Open Chrome or any browser at `http://localhost:3000`.

---

### Step 2: Overview Dashboard Tour (2 Minutes)
1. **Navigate to the Overview Tab:**
   - Point out the **Live System Ready indicator** in the upper-right corner.
   - Explain the 6 KPI Cards:
     - *Observed Vehicles:* Total vehicle tracks counted across all camera streams.
     - *Total Violations:* Real database records (pending vs confirmed).
     - *No Helmet, Multiple Riders, Wrong-Way:* Sub-category breakdown.
     - *Active Pipelines:* Real-time indicator of queued video jobs.
2. **Highlight the Interactive Charts:**
   - Hover over the **Violation Category Distribution Pie Chart** (Recharts) showing live proportional breakdown.
   - Show the **Infractions by Camera Location Bar Chart**.
3. **Recent Violations Ledger:**
   - Show the live table of recent alerts at the bottom of the page.

---

### Step 3: Media Upload & Live Video Processing (3 Minutes)
1. **Navigate to the "Upload & Analysis" Tab:**
   - Show the drag-and-drop zone.
   - Explain the **Feed & Rule Parameters Panel** on the right:
     - Target Camera Stream dropdown (`CAM-01: North Junction Highway Approach`).
     - Detection Confidence slider (e.g., 35%).
     - Motorcycle Passenger Limit selector (default: 2).
2. **Execute Video Analysis:**
   - Click the file upload box and select `sample_media/traffic_video.mp4` (or drag & drop it).
   - Click **"Execute Analysis"**.
   - **Show the Live Progress Bar:** Note how the percentage smoothly ticks from 0% to 100% with the exact frame count displayed (`25/25 frames`).
   - Mention to the faculty: *"Notice how the HTTP connection was non-blocking. A background task in FastAPI processed the video while the frontend polled the job status."*
3. **Inspect the Processed Results:**
   - Once completed, the video player renders the **annotated MP4 video** showing vehicle bounding boxes, tracking IDs, and the red violation overlay on the opposing vehicle!
   - Show the **Violations Generated During This Run** cards at the bottom.

---

### Step 4: Evidentiary Audit & Human-in-the-Loop Review (2 Minutes)
1. Click **"Inspect Evidence"** on the generated violation card (or switch to the **"Violation Records"** tab).
2. The **Evidence Inspection Modal** pops up:
   - Toggle between **"Annotated Evidence"**, **"Original Frame"**, and **"Vehicle Crop"** using the top pill buttons.
   - Show the **AI Analytical Explanation** box: Point out how it provides the exact mathematical reasoning (e.g. *"Vehicle Track #2 driving against legal direction. Observed movement heading 270.0° vs legal heading 90.0°... confirmed over consecutive frames with displacement"*).
3. **Demonstrate Human Review Decision Support:**
   - Click **"Confirm Violation"** (turns green, changes database status).
   - Explain: *"The AI does not unilaterally issue fines; it flags candidates and prepares evidentiary dossiers for operator sign-off."*

---

### Step 5: Camera Management & 360° Compass Calibration (2 Minutes)
1. **Navigate to the "Camera Management" Tab:**
   - Show the configured cameras (`CAM-01`, `CAM-02`).
   - Point out the **visual compass dial** showing the legal flow vector.
2. **Add a New Camera:**
   - Click **"Add Camera Feed"**.
   - Enter ID: `CAM-03`, Name: `East Expressway Toll Plaza`, Location: `KM 28 Eastbound`.
   - **Click on the interactive compass dial:** Click on East (90°) or drag the slider to 0°. Note how the heading angle dynamically updates to `0° Heading`.
   - Toggle rules (Helmet, Multiple Riders, Wrong-Way).
   - Click **"Save Camera Configuration"**. The new camera immediately renders in the dashboard!

---

### Step 6: Model Status & Scientific Integrity Disclosure (2 Minutes)
1. **Navigate to the "Model Status" Tab:**
   - Show the **Host Hardware & Telemetry** (live CPU usage, RAM consumption, disk space, MPS/GPU status).
   - Point out the **Active Vehicle Detector:** YOLOv8 nano online with 6 target classes.
2. **Address the Helmet Detector Honestly:**
   - Point to the amber banner: *"Notice that the system explicitly states that dedicated helmet weights are not loaded, so helmet violation generation is safely paused."*
   - Explain to faculty: *"Many student projects generate fake bounding boxes when models are missing. In contrast, our system maintains rigorous academic integrity: we report model status honestly and require genuine weights before making legal determinations."*
   - Show the code in `ml/training/train_helmet.py` and `ml/datasets/dataset_prep.py` where custom training is configured.

---

### Step 7: Automated Test Verification (1 Minute)
1. Open terminal and run:
   ```bash
   ./venv/bin/pytest -v
   ```
2. Show the evaluators that **all 14 test modules pass in ~3.3 seconds**, covering tracking, association geometry, API integration, and end-to-end video analysis.
3. Conclude the demonstration and invite questions.

