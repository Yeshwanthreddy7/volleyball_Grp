# AGENT – What This Agent Is Working On

## Project: Volleyball Tactical Analysis with Mamba SSM

This document describes the goal, approach, and design decisions of the
AI agent building this final-year project (FYP).

---

## Problem Statement

Volleyball coaching staff need fast, consistent feedback on team tactics
during a match or in post-game review.  The full pipeline transforms raw
match video into a final tactical coordination report:

```
Match Video (.mp4)
      ↓
Frame Extraction
      ↓
Player Detection  (YOLOv8)
      ↓
Player Tracking   (ByteTrack)
      ↓
Pose / Position Extraction
      ↓
Feature Engineering  (velocity, spacing, centroid, sync_score)
      ↓
Sequence Formation  (29-frame time-series windows)
      ↓
Mamba Model  (Sequence Learning)
      ↓
Coordination Classification
      ↓
Final Tactical Report
```

---

## What the Agent Has Built

### 1. `pipeline.py` – End-to-End Pipeline *(new)*

Implements the complete processing chain from a raw match video to a
printed Team Coordination Analysis report.

**Key stages:**

| Stage | Implementation |
|-------|---------------|
| Frame Extraction | `cv2.VideoCapture` |
| Player Detection | `ultralytics.YOLO` (YOLOv8n, person class) |
| Player Tracking | `supervision.ByteTrack` |
| Position Extraction | Bounding-box centroids → court cm via homography or linear scale |
| Feature Engineering | `_compute_frame_features` (14-dim row); `_sequence_metrics` (sync, spacing, centroid vel) |
| Sequence Formation | Non-overlapping 29-frame windows buffered in memory |
| Mamba Classification | `_classify_sequence` using loaded checkpoint |
| Tactical Report | `generate_report` (event counts + efficiency score) |

**Court coordinate handling:**

- Default: pixel coords scaled linearly to 1800 × 900 cm
- Optional: perspective homography from four user-supplied corner points
  (`--court-corners TL TR BR BL`) for accurate court mapping

**Per-sequence metrics output:**

```
sync_score               – mean pairwise cosine similarity of player velocity vectors
mean_spacing_cm          – average nearest-neighbour player spacing
centroid_vel_cm_per_frame – team centroid movement speed
```

### 2. `mamba_model.py` – Architecture

Implements a **pure-PyTorch Mamba SSM** classifier from scratch,
following the original paper (Gu & Dao, arXiv:2312.00752).

### 3. `train_mamba.py` – Training Pipeline

Loads labelled CSVs from `label_clips.py`, trains with AdamW + cosine
annealing, saves best checkpoint by validation accuracy.

### 4. `infer_mamba.py` – Inference Script

Loads a saved checkpoint, processes a directory of raw clip CSVs, and
now prints a **Team Coordination Analysis** report with numeric label
indices and coordination efficiency score.

### 5. `label_clips.py` – Rule-Based Labeller *(extended)*

Added:
- **`_compute_sync_score(positions)`** – movement synchronization metric
  (mean pairwise cosine similarity of velocity vectors)
- **`generate_report(label_counts)`** – final report with per-label counts
  and coordination efficiency score
- **`LABEL_INDEX`** / **`LABEL_TO_INDEX`** – numeric ↔ string label mapping
- `process_directory()` now prints the final report after processing

### 6. `requirements.txt` – Updated Dependencies

Added `ultralytics>=8.0.0` and `supervision>=0.18.0` alongside the
existing OpenCV / NumPy / pandas / PyTorch stack.

### 7. `README.md` – Full Documentation

Completely rewrote the README to cover:
- Full pipeline overview (diagram)
- Module table
- Tactical labels with numeric indices
- Example final report output
- Key parameters table
- Step-by-step usage guide (pipeline → label → train → infer)
- Data format specification
- Model architecture diagram

---

## Output Labels (numeric index → string)

| Index | Label |
|-------|-------|
| 1 | Coordinated Attack |
| 2 | Coordinated Defense |
| 3 | Delayed Support |
| 4 | Spacing Breakdown |

## Final Report Format

```
============================================
       Team Coordination Analysis
============================================
  [1] Coordinated attack events  : 18
  [2] Coordinated defense events : 4
  [3] Delayed support events     : 7
  [4] Spacing breakdown events   : 5

  Coordination efficiency score  : 65%
============================================
```

---

## Data Flow

```
Raw MP4
  │
  ▼
pipeline.py  ──────────────────────────────────────────────────────────────────
  │  Frame Extraction (cv2)
  │  → Player Detection (YOLOv8)
  │  → Player Tracking (ByteTrack)
  │  → Position Extraction (bounding-box centroids → court cm)
  │  → Feature Engineering (velocity, spacing, centroid, sync_score)
  │  → Sequence Formation (29-frame windows)
  │  → Mamba Classification (loaded checkpoint)
  │  → Annotated MP4  +  Predictions CSV  +  Final Tactical Report
  │
  ▼  (alternative: external tracking system produces CSVs)
  │
label_clips.py  →  Labelled CSVs  (frames 1–29 + target_label)
  │                               ↑  rule-based labels used as ground truth
  ▼
train_mamba.py  →  mamba_checkpoint.pt
  │
  ▼
infer_mamba.py  →  Predictions for new clips  +  Final Tactical Report
```

---

## Current Limitations & Future Work

| Limitation | Planned Fix |
|------------|-------------|
| Sequential scan is O(L·D·N) – slow for very long sequences | Replace with parallel scan or use `mamba-ssm` package when CUDA is available |
| Training data comes from rule-based labels (teacher noise) | Annotate a gold-standard validation set with human coaches |
| Only 4 tactical classes | Extend label set with finer-grained patterns (e.g. block timing, setter coverage) |
| No data augmentation | Add coordinate jitter, temporal sub-sampling, mirroring |
| YOLOv8 uses general COCO weights | Fine-tune on volleyball footage for better ball/player precision |

---

## References

- Gu, A., & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* arXiv:2312.00752.
- Gu, A., Goel, K., & Ré, C. (2022). *Efficiently Modeling Long Sequences with Structured State Spaces.* ICLR 2022.
- Zhang, Y. et al. (2022). *ByteTrack: Multi-Object Tracking by Associating Every Detection Box.* ECCV 2022.
