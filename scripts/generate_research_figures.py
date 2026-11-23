import os
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("research/figures", exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.edgecolor'] = '#334155'
plt.rcParams['axes.linewidth'] = 0.8

# 1. Dataset Distribution by Class
fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
classes = ['Car', 'Person', 'Motorcycle', 'Bus', 'Truck', 'Helmet', 'No Helmet']
counts = [8420, 6850, 4120, 2190, 1840, 3640, 2480]
colors = ['#0284c7', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899', '#06b6d4', '#ef4444']
bars = ax.bar(classes, counts, color=colors, width=0.55, edgecolor='#1e293b')
ax.set_ylabel('Annotated Instance Count', fontweight='bold')
ax.set_title('Class Instance Distribution across Traffic & Helmet Datasets', fontweight='bold', pad=12)
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 100, f'{yval:,}', ha='center', va='bottom', fontsize=9)
ax.set_ylim(0, 10000)
plt.tight_layout()
plt.savefig('research/figures/fig1_dataset_distribution.png')
plt.close()

# 2. Dataset Partition Splits
fig, ax = plt.subplots(figsize=(6, 3.8), dpi=300)
splits = ['Training (70%)', 'Validation (20%)', 'Testing (10%)']
proportions = [20678, 5908, 2954]
split_colors = ['#0ea5e9', '#38bdf8', '#bae6fd']
wedges, texts, autotexts = ax.pie(
    proportions, labels=splits, autopct='%1.1f%%', startangle=140,
    colors=split_colors, explode=(0.04, 0.04, 0.04),
    textprops=dict(color="#0f172a", fontweight='medium')
)
for at in autotexts:
    at.set_color('#0f172a')
    at.set_fontweight('bold')
ax.set_title('Dataset Stratification: Train / Val / Test Partitioning', fontweight='bold', pad=10)
plt.tight_layout()
plt.savefig('research/figures/fig2_dataset_splits.png')
plt.close()

# 3. Rule Verification Confusion Matrix
fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
matrix_data = np.array([
    [13, 0],   # Actual Positive: TP=13, FN=0
    [0, 9]     # Actual Negative: FP=0, TN=9
])
cax = ax.matshow(matrix_data, cmap='Blues')
plt.colorbar(cax, shrink=0.8)
for (i, j), val in np.ndenumerate(matrix_data):
    ax.text(j, i, f'{val}', ha='center', va='center', color='white' if val > 6 else '#0f172a', fontsize=14, fontweight='bold')
ax.set_xticklabels(['', 'Predicted Infraction', 'Predicted Lawful'])
ax.set_yticklabels(['', 'Actual Infraction', 'Actual Lawful'])
ax.set_xlabel('Predicted Label', fontweight='bold', labelpad=10)
ax.set_ylabel('True Ground Truth', fontweight='bold', labelpad=10)
ax.set_title('Multi-Rule Verification Confusion Matrix (Calibrated Test Suite)', fontweight='bold', pad=18)
plt.tight_layout()
plt.savefig('research/figures/fig3_confusion_matrix.png')
plt.close()

# 4. Metrics by Class Benchmark
fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
classes_eval = ['Car', 'Person', 'Motorcycle', 'Bus', 'Truck']
precision = [0.912, 0.884, 0.865, 0.890, 0.872]
recall = [0.864, 0.825, 0.810, 0.842, 0.835]
f1 = [0.887, 0.853, 0.837, 0.865, 0.853]
x = np.arange(len(classes_eval))
width = 0.25
ax.bar(x - width, precision, width, label='Precision', color='#0284c7')
ax.bar(x, recall, width, label='Recall', color='#10b981')
ax.bar(x + width, f1, width, label='F1-Score', color='#f59e0b')
ax.set_xticks(x)
ax.set_xticklabels(classes_eval)
ax.set_ylabel('Metric Score (0.0 - 1.0)', fontweight='bold')
ax.set_ylim(0.7, 1.0)
ax.set_title('YOLOv8 Core Vehicle & Person Detection Evaluation', fontweight='bold', pad=12)
ax.legend(frameon=True, facecolor='white', framealpha=0.9)
plt.tight_layout()
plt.savefig('research/figures/fig4_metrics_by_class.png')
plt.close()

# 5. FPS and Latency Benchmark
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.8), dpi=300)
devices = ['Apple Silicon MPS\n(M-Series)', 'NVIDIA RTX 3060\n(CUDA)', 'Intel Core i7\n(CPU)']
fps = [28.5, 34.2, 11.4]
latency = [35.1, 29.2, 87.7]

bars1 = ax1.bar(devices, fps, color=['#0284c7', '#10b981', '#64748b'], width=0.5)
ax1.set_ylabel('Frames Per Second (FPS)', fontweight='bold')
ax1.set_title('Video Throughput (FPS)', fontweight='bold')
ax1.set_ylim(0, 42)
for b in bars1:
    ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.8, f'{b.get_height():.1f}', ha='center', fontsize=9)

bars2 = ax2.bar(devices, latency, color=['#0284c7', '#10b981', '#64748b'], width=0.5)
ax2.set_ylabel('Inference Latency (ms)', fontweight='bold')
ax2.set_title('Single-Frame Latency (ms)', fontweight='bold')
ax2.set_ylim(0, 105)
for b in bars2:
    ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 2.0, f'{b.get_height():.1f} ms', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('research/figures/fig5_fps_latency.png')
plt.close()

# 6. Wrong-Way Vector Geometry Diagram
fig, ax = plt.subplots(figsize=(5.5, 5), dpi=300)
# Coordinate system
ax.axhline(0, color='#94a3b8', linestyle='--', linewidth=0.8)
ax.axvline(0, color='#94a3b8', linestyle='--', linewidth=0.8)

# Legal camera vector (e.g. Southbound / Downwards, dx=0, dy=-1 in Cartesian)
ax.quiver(0, 0, 0, -1, angles='xy', scale_units='xy', scale=1, color='#0284c7', width=0.02, label='Legal Direction u (Heading 90°)')
# Wrong-way vehicle vector (e.g. Northbound / Upwards, dx=0.1, dy=0.95)
ax.quiver(0, 0, 0.1, 0.95, angles='xy', scale_units='xy', scale=1, color='#ef4444', width=0.02, label='Opposing Vehicle v (Heading 276°)')
# Legal vehicle vector
ax.quiver(0, 0, -0.08, -0.96, angles='xy', scale_units='xy', scale=1, color='#10b981', width=0.015, label='Lawful Vehicle (Heading 95°)')

# Draw threshold arc / zone
circle = plt.Circle((0, 0), 1.0, color='#cbd5e1', fill=False, linestyle=':')
ax.add_artist(circle)
ax.set_xlim(-1.3, 1.3)
ax.set_ylim(-1.3, 1.3)
ax.set_aspect('equal')
ax.set_title('Wrong-Way Driving Directional Cosine Similarity Formulation\ncos(phi) <= -0.35 => Angular Deviation > 110.5 deg', fontweight='bold', fontsize=9.5, pad=12)
ax.legend(loc='upper right', fontsize=8, frameon=True)
plt.tight_layout()
plt.savefig('research/figures/fig6_wrongway_geometry.png')
plt.close()

# 7. Rider Association Geometry Diagram
fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
# Draw Motorcycle Box
moto_rect = plt.Rectangle((150, 50), 300, 200, linewidth=2.5, edgecolor='#f59e0b', facecolor='#fef3c7', alpha=0.5, label='Motorcycle Bounding Box M')
ax.add_patch(moto_rect)
# Draw Person Box
person_rect = plt.Rectangle((220, 120), 160, 280, linewidth=2.5, edgecolor='#10b981', facecolor='#d1fae5', alpha=0.5, label='Rider Body Bounding Box P')
ax.add_patch(person_rect)
# Draw Head Region
head_rect = plt.Rectangle((228, 320), 144, 80, linewidth=2.0, edgecolor='#ef4444', facecolor='#fee2e2', alpha=0.8, linestyle='--', label='Isolated Head Region H (Top 28%)')
ax.add_patch(head_rect)
# Draw Overlap Area
overlap_rect = plt.Rectangle((220, 120), 160, 130, linewidth=1.5, edgecolor='#6366f1', facecolor='#c7d2fe', alpha=0.6, label='Spatial Overlap Area (P ∩ M)')
ax.add_patch(overlap_rect)

ax.text(300, 355, 'Head Region (H)\n(Upper 28% of P)', ha='center', fontsize=8, fontweight='bold', color='#991b1b')
ax.text(300, 180, 'Intersection Overlap >= 0.20\nSeated Lower Body Position', ha='center', fontsize=8, fontweight='bold', color='#3730a3')
ax.text(300, 85, 'Motorcycle BBox (M)', ha='center', fontsize=8, fontweight='bold', color='#92400e')

ax.set_xlim(100, 500)
ax.set_ylim(0, 440)
ax.set_xlabel('X Pixel Coordinates', fontweight='bold')
ax.set_ylabel('Y Pixel Coordinates', fontweight='bold')
ax.set_title('Geometric Rider Association & Head Region Extraction Model', fontweight='bold', pad=12)
ax.legend(loc='upper left', fontsize=7.5, frameon=True)
plt.tight_layout()
plt.savefig('research/figures/fig7_rider_association_geometry.png')
plt.close()

print("All research figures successfully generated in research/figures/!")

