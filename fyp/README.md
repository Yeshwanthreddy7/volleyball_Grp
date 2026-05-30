# FYP – Volleyball Analytics with Mamba SSM

An end-to-end volleyball play analysis system that combines
**YOLOv8 player detection**, **ByteTrack multi-object tracking**, and a
**Mamba selective state-space model** to automatically label tactical
coordination patterns from raw match video.

---

## Pipeline Overview

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

## Modules

| Module | Purpose |
|--------|---------|
| `pipeline.py` | **Full end-to-end pipeline** (video → report) |
| `process_video.py` | Lightweight video annotator (MOG2 background subtraction) |
| `label_clips.py` | Rule-based auto-labeller for 41-frame tracking CSVs |
| `mamba_model.py` | Pure-PyTorch Mamba SSM classifier architecture |
| `train_mamba.py` | Train the Mamba model on labelled clip CSVs |
| `infer_mamba.py` | Run a trained Mamba model on pre-extracted CSV clips |

---

## Background: Why Mamba?

Traditional rule-based labellers encode fixed thresholds that may not
generalise across different teams or courts.  **Mamba** (Gu & Dao, 2023)
is a selective state-space sequence model that:

- Processes sequences in **linear time** (no quadratic attention overhead)
- Learns **selective** memory: which past frames to remember vs. forget
- Outperforms Transformers on long time-series benchmarks
- Is well-suited to the fixed-length (29-frame) coordinate sequences in this project

---

## Tactical Labels

| Index | Label | Description |
|-------|-------|-------------|
| **1** | **Coordinated Attack** | Two fast attackers + stationary base players near the net |
| **2** | **Coordinated Defense** | Entire team shifting as a rigid unit |
| **3** | **Delayed Support** | Closest player reacts too late after ball impact |
| **4** | **Spacing Breakdown** | Players too spread or too clustered (structural failure) |

---

## Output: Final Tactical Report

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

The **coordination efficiency score** = fraction of sequences classified as
Coordinated Attack or Coordinated Defense × 100.

---

## Key Parameters Used to Generate Labels

| Parameter | Formula | Used by |
|-----------|---------|---------|
| **Player Spacing** | `distance(player_i, player_j)` for all 15 pairs | Rule 1 – Spacing Breakdown |
| **Movement Synchronization** | `sync_score = mean cosine_similarity(velocity_vectors)` | Metrics output |
| **Player Arrival Timing** | `arrival_time_diff = \|t_A − t_B\|` (frames to peak speed) | Rule 2 – Delayed Support |
| **Team Centroid Movement** | `centroid = mean(player_positions)` per frame | Rule 4 – Coordinated Defense |

---

## Getting Started

### Prerequisites

```
Python >= 3.9
```

### Installation

```bash
pip install -r requirements.txt
```

> **GPU users** – install the CUDA build of PyTorch first:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

---

## Usage

### End-to-End Pipeline (recommended)

Run the complete pipeline from a raw match video to a final tactical report:

```bash
python pipeline.py match.mp4 mamba_checkpoint.pt \
    --output-video annotated.mp4 \
    --output-csv   predictions.csv
```

With accurate court mapping via perspective homography (supply the four
pixel corners of the court boundary in TL→TR→BR→BL order):

```bash
python pipeline.py match.mp4 mamba_checkpoint.pt \
    --court-corners 42,18 1238,18 1238,702 42,702 \
    --output-video  annotated.mp4 \
    --output-csv    predictions.csv
```

| Flag | Default | Description |
|------|---------|-------------|
| `--yolo-model` | `yolov8n.pt` | YOLOv8 weights (downloads automatically) |
| `--conf-threshold` | `0.35` | YOLOv8 detection confidence threshold |
| `--court-corners` | *(none)* | 4 pixel corners for homography (TL TR BR BL) |
| `--output-video` | *(none)* | Path for annotated output video |
| `--output-csv` | *(none)* | Path for per-sequence predictions CSV |

#### Per-sequence output CSV columns

| Column | Description |
|--------|-------------|
| `sequence` | Sequence number (1-based) |
| `frame_start` / `frame_end` | Frame range covered |
| `label` | Predicted label string |
| `label_index` | Numeric label index (1–4) |
| `confidence` | Softmax probability of predicted class |
| `sync_score` | Movement synchronization score (–1 to +1) |
| `mean_spacing_cm` | Average nearest-neighbour player spacing (cm) |
| `centroid_vel_cm_per_frame` | Team centroid velocity |
| `p_<label>` | Per-class softmax probabilities |

---

### Step-by-Step (alternative workflow)

#### 1 – Annotate a video (lightweight, no model needed)

```bash
python process_video.py test1.mp4 output_annotated.mp4
```

#### 2 – Generate labelled training data (rule-based)

```bash
python label_clips.py ./clips_directory
```

Processes every CSV in `./clips_directory`. Each CSV must contain
columns `frame_id, ball_x, ball_y, p1_x, p1_y, …, p6_x, p6_y`.

- Evaluation window (frames 30–41) → determines the label
- Training window (frames 1–29) + `target_label` column → overwrites the CSV
- Unclassified clips are deleted
- Prints a **Team Coordination Analysis** report at the end

#### 3 – Train the Mamba model

```bash
python train_mamba.py ./clips_directory \
    --epochs 50 \
    --batch_size 32 \
    --d_model 64 \
    --n_layers 4 \
    --checkpoint mamba_checkpoint.pt
```

Key options:

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 50 | Training epochs |
| `--batch_size` | 32 | Mini-batch size |
| `--d_model` | 64 | Mamba model dimension |
| `--n_layers` | 4 | Number of stacked Mamba blocks |
| `--d_state` | 16 | SSM state dimension |
| `--d_conv` | 4 | Depthwise conv kernel width |
| `--lr` | 1e-3 | Initial learning rate (cosine decay) |
| `--val_split` | 0.2 | Fraction of data held out for validation |
| `--checkpoint` | `mamba_checkpoint.pt` | Output checkpoint path |

#### 4 – Classify new clips

```bash
python infer_mamba.py mamba_checkpoint.pt ./new_clips \
    --write \
    --output_csv predictions.csv
```

| Flag | Description |
|------|-------------|
| `--write` | Overwrite each input CSV with the predicted `target_label` column |
| `--output_csv` | Save a summary table of all predictions |

---

## Data Format

Each clip CSV must follow this schema:

```
frame_id, ball_x, ball_y, p1_x, p1_y, p2_x, p2_y, p3_x, p3_y,
          p4_x, p4_y, p5_x, p5_y, p6_x, p6_y
```

- **Court coordinate system**: 1800 × 900 cm top-down view, net at Y = 0
- **Frame rate**: 30 FPS
- **Clip length**: 41 frames (frames 1–29 used for training/inference,
  frames 30–41 used for rule-based label evaluation)

---

## Model Architecture

```
Input (29, 14)
    │
    ▼
Linear Embed → (29, d_model)
    │
    ▼  ×n_layers
┌─────────────────────────────┐
│ LayerNorm                   │
│ SelectiveSSM (Mamba Block)  │
│  ├─ Input projection        │
│  ├─ Depthwise Conv-1D       │
│  ├─ SSM (A, B, C, Δ)        │
│  └─ Gated output            │
│ Residual add                │
└─────────────────────────────┘
    │
    ▼
LayerNorm → Global Avg Pool → (d_model,)
    │
    ▼
Linear → GELU → Linear → (4 classes)
```

**SSM parameters** (all learnable except A which is initialised and fine-tuned):

| Symbol | Shape | Role |
|--------|-------|------|
| A | (D, N) | State transition (fixed diagonal, log-space) |
| B | (B, L, N) | Input-to-state (input-dependent, selective) |
| C | (B, L, N) | State-to-output (input-dependent, selective) |
| Δ | (B, L, D) | Time step / discretisation (input-dependent) |

---

## File Structure

```
fyp/
├── pipeline.py            # End-to-end pipeline (video → tactical report)
├── process_video.py       # Lightweight video annotation (MOG2)
├── label_clips.py         # Rule-based clip labeller (data preparation)
├── mamba_model.py         # Mamba SSM classifier (architecture)
├── train_mamba.py         # Model training script
├── infer_mamba.py         # Inference / prediction script
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── AGENT.md               # Agent working description
└── test1.mp4              # Sample input video
```

---

## Reference

Gu, A., & Dao, T. (2023).
**Mamba: Linear-Time Sequence Modeling with Selective State Spaces.**
*arXiv:2312.00752*. https://arxiv.org/abs/2312.00752
