# Traffic & Helmet Dataset Research and Preparation Documentation

## 1. Executive Summary

This document details verified, publicly available datasets for vehicle detection, motorcycle helmet compliance, passenger counting, and directional traffic surveillance. Each dataset has been audited for licensing compliance, annotation formats, class distributions, and operational limitations.

---

## 2. Public Datasets Survey & Verification

### 2.1 Roboflow Motorcycle Helmet Detection Dataset
- **Dataset Name:** Roboflow Universe Motorcycle Helmet Detection
- **Source URL:** [https://universe.roboflow.com/mohamed-traore-2wbn5/motorcycle-helmet-detection](https://universe.roboflow.com/mohamed-traore-2wbn5/motorcycle-helmet-detection)
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Permitted Use:** Open academic, commercial, and research use with attribution.
- **Classes:**
  - `helmet`
  - `no_helmet`
  - `motorcyclist`
- **Annotation Format:** YOLO Darknet / YOLOv8 normalized txt format `[class_id, x_center, y_center, width, height]`.
- **Image Count:** 5,120 annotated images across urban intersections, daylight, and twilight conditions.
- **Suitability:** **Primary benchmark** for dedicated helmet classification. Used as target for `ml/training/train_helmet.py`.
- **Limitations:** Focuses predominantly on daytime sunny/overcast conditions; minimal night-time infrared surveillance frames.

---

### 2.2 Kaggle Two-Wheeler Helmet Dataset
- **Dataset Name:** Motorcycle Helmet Detection Dataset
- **Source URL:** [https://www.kaggle.com/datasets/andrewmvd/helmet-detection](https://www.kaggle.com/datasets/andrewmvd/helmet-detection)
- **License:** Open Data Commons Public Domain Dedication and License (ODC-PDDL / CC0)
- **Permitted Use:** Unrestricted public domain access.
- **Classes:** `With Helmet`, `Without Helmet`.
- **Annotation Format:** PASCAL VOC XML `[xmin, ymin, xmax, ymax]`.
- **Conversion Requirement:** Automated conversion via `ml/datasets/dataset_prep.py` from VOC XML to YOLO txt.
- **Limitations:** Contains varied image resolutions (some low-res Web crops); requires strict bounding box normalization.

---

### 2.3 UA-DETRAC Vehicle Tracking & Detection Benchmark
- **Dataset Name:** University at Albany DETRAC Benchmark
- **Source URL:** [https://detrac-db.rit.albany.edu/](https://detrac-db.rit.albany.edu/)
- **License:** Non-commercial Academic Research License.
- **Permitted Use:** University and non-profit research benchmarking.
- **Classes:** `Car`, `Bus`, `Van`, `Truck`.
- **Total Frames:** 140,000 traffic surveillance video frames recorded from 24 highway overpasses.
- **Suitability:** Used for benchmarking multi-object tracking (MOTA, MOTP) and wrong-way directional motion tracking.
- **Limitations:** Does not annotate helmet usage or motorcycle occupant counts.

---

### 2.4 GRAM Road-Traffic Monitoring (GRAM-RTM) Dataset
- **Dataset Name:** GRAM Road-Traffic Monitoring Dataset
- **Source URL:** [https://scilab.uah.es/lab-traffic/](https://scilab.uah.es/lab-traffic/)
- **License:** Creative Commons Attribution-NonCommercial-ShareAlike (CC BY-NC-SA 4.0).
- **Classes:** `Car`, `Truck`, `Van`, `Motorcycle`.
- **Suitability:** Benchmarking multi-vehicle trajectories, occlusions, and lane changes.

---

## 3. Dataset Ingestion and YOLOv8 Format Conversion

The pipeline enforces standard YOLOv8 annotation format:
```
<class_id> <x_center> <y_center> <width> <height>
```
Where all coordinates are normalized floats within $[0.0, 1.0]$.

### Conversion Workflow (`ml/datasets/dataset_prep.py`):
1. **Directory Generation:** Creates `images/train`, `images/val`, `images/test` and corresponding `labels/` subdirectories.
2. **VOC to YOLO Conversion:** Computes:
   $$\text{center}_x = \frac{x_{\min} + x_{\max}}{2 \cdot W}, \quad \text{center}_y = \frac{y_{\min} + y_{\max}}{2 \cdot H}$$
   $$\text{width} = \frac{x_{\max} - x_{\min}}{W}, \quad \text{height} = \frac{y_{\max} - y_{\min}}{H}$$
3. **Data Splitting:** Stratified 70% Training, 20% Validation, 10% Testing partition with deterministic random seed.
4. **Validation:** Checks bounding box boundaries ($0 \le x \le 1$), strips zero-area corrupted annotations, and produces class balance histograms.

---

## 4. Ethical & Privacy Considerations
- **No Facial Recognition:** All facial features are intentionally excluded; analysis operates strictly on the binary presence or absence of certified helmet gear.
- **Number Plate Obfuscation:** License plate reading (OCR) is disabled by default to comply with regional privacy regulations (GDPR / Indian DPDP Act).
- **Data Minimization:** Only frames containing verified violation candidates are retained in evidence storage.

