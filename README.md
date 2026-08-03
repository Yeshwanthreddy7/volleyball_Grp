# Automated Volleyball Tactical Analysis from Broadcast Video

Final Year Project — research-based. Converts raw broadcast match video into a
per-rally tactical coordination report, using detection → tracking → court
projection → identity-stable formation features → a learned sequence classifier.

**Everything in this README is reproducible with one command:**

```bash
python reproduce.py
```

It regenerates every metric and figure quoted below into `results/latest/`,
records package versions and SHA-256 hashes of every model file used, asserts
the properties the report claims, and **exits non-zero if any of them fail.**

---

## 1. Problem and research gap

Coaching staff need consistent, fast feedback on team coordination. Manual
notation is slow and subjective; existing automated sports analytics focus on
*individual* actions (spike, block, serve) or on ball trajectory, not on
**whole-team spatial coordination**.

The gap this project addresses: classifying a rally window by the *collective*
tactical pattern of six players — how the formation is shaped, how it moves, and
whether it moves together — from a single uncalibrated broadcast camera.

**Four tactical classes**, plus a class-0 noise sink for dead-ball / transition
windows:

| Index | Class | Definition |
|---|---|---|
| 1 | Coordinated Attack | fast approach by front-court attackers with base players held near the net |
| 2 | Coordinated Defense | the team shifts as a rigid unit |
| 3 | Delayed Support | the nearest player reacts late after ball impact |
| 4 | Spacing Breakdown | structural failure — formation too spread or too clustered |
| 0 | Unclassified / Transition | dead ball, too few players, out of play |

---

## 2. Architecture

```
broadcast video (1280x720, 30 fps)
   │
   ├─ PREFLIGHT GATE ─────────── samples the segment, measures the surviving
   │                             player population per stage, ABORTS (exit 2)
   │                             if it is degenerate                fyp/preflight.py
   ▼
DETECTION            dual backend                                  fyp/interfaces.py
   ├─ players: stock YOLOv11 (COCO person) @ imgsz 1280
   └─ ball   : volleyball fine-tune        @ imgsz 1280
   ▼
COURT CALIBRATION    auto court-quad detection, camera-motion            fyp/court.py
                     compensation, binary in/out-of-court mask
   ▼
TRACKING             BoT-SORT / ByteTrack + Kalman tuned for          fyp/tracking.py
                     explosive kinetics; occlusion-bridged slots      fyp/identity.py
   ▼
TEAM SEPARATION      jersey-colour clustering, clusters labelled by       fyp/teams.py
                     median court-side, per-track majority vote
   ▼
FEATURES             18-dim permutation-invariant per-frame           fyp/features.py
                     descriptor (`perm_invariant_v2`)
   ▼
WINDOWING            non-overlapping 29-frame (≈1 s) windows
   ▼
CLASSIFIER           learned — either backend, chosen by file extension
   ├─ scikit-learn (default)                                 fyp/ml_classifier.py
   └─ Mamba SSM / Transformer                                 fyp/mamba_model.py
   ▼
REPORT               per-sequence CSV + annotated MP4 + coordination report
```

Both classifier backends sit behind one `BaseTemporalClassifier` interface and
consume the identical `(29, 18)` tensor, so swapping them is a filename change.

---

## 3. Method — the parts that required a decision

### 3.1 Dual detector

A single detector cannot serve both classes. Measured on this project's own
footage (`results/latest/detector_ablation.csv`):

- **Player** is a canonical COCO object. Stock `yolo11n.pt` returns 14–23
  people/frame on all three source videos. The 416-image Roboflow fine-tune
  collapses to **0.3 players/frame** on unseen footage — max class confidence
  0.06 on a frame containing twelve clearly visible players.
- **Ball** is a ~15 px motion-blurred object. Here the fine-tune wins decisively:
  **67% recall vs stock's 16%** on the same video.

So players come from stock weights and the ball from the fine-tune.

**Inference resolution is not a detail.** A volleyball spans ~15–20 px on a 720p
frame; the Ultralytics default of 640 downscales it to ~9 px, below the smallest
detection stride. Ball recall with *identical weights*: **15% at 640 → 77% at
1280**.

### 3.2 Team separation by jersey colour

The court mask removes coaches, referees and crowd, but still passes **both
teams** — ten players into a six-slot system. No foot-point-vs-net rule can fix
this: at a block, both front rows stand within a metre of each other, on opposite
sides of a line a few pixels wide in a foreshortened view.

Two signals, each used only where it is reliable:

- **Geometry is reliable in aggregate.** Players may not cross the net, so a
  colour cluster's *median* court-side over a warm-up window is a solid team label.
- **Colour is reliable instantaneously**, but has no intrinsic team meaning.

Cluster torso colour → label each cluster by its members' median court-side →
decide membership by colour thereafter, majority-voted per track.
`n_clusters=4` by default because FIVB rules require a contrasting libero
jersey, so a team is *two* colour populations, not one.

Measured: **10.7 → 5.8 players/frame** (team size is 6).

### 3.3 Scale-invariant features

Without a per-camera homography the pipeline uses a **linear** pixel→court
scaling, so "cm" means something different in every video. Absolute features
(`hull_area`, `centroid_x`, `max_pair_dist`) are therefore partly *camera
identity*, and a learner will use them to recognise which video it is looking at.

The invariant block feeds only quantities that survive an unknown scale factor:
dimensionless ratios (compactness, tightest/loosest pair, speed ratio) and the
per-clip z-scored temporal *shape* of every channel.

### 3.4 Evaluation protocol

Two protocols are always reported together:

- **Stratified k-fold** — clips from one video land on both sides. Measures
  *within-video* generalisation. Optimistic.
- **Leave-one-video-out (LOVO)** — every clip of a held-out video is unseen.
  Measures *cross-video* generalisation. This is the honest number.

**The gap between them is a result, not an embarrassment** — it quantifies how
much the model relies on camera-specific cues.

---

## 4. Results

Regenerate with `python reproduce.py`; full tables in `results/latest/METRICS.md`.

Headline (285 clips, 3 source videos, scale-invariant features):

| | k-fold macro-F1 | LOVO macro-F1 | LOVO kappa |
|---|---|---|---|
| absolute features | **0.520** | 0.265 | 0.081 |
| scale-invariant features | 0.436 | **0.297** | **0.157** |

Read the *direction*: dimensionless features **lower** the within-video score and
**raise** the cross-video score, nearly doubling kappa. That is the signature of
removing a shortcut, and it is the strongest single finding in the evaluation.

### Honest limitations — state these, do not hide them

1. **Labels come from a rule engine** (`fyp/label_clips.py`). Cross-validation
   therefore measures *agreement with a heuristic teacher*, **not** tactical
   correctness. Section 6 below is the protocol that fixes this.
2. **Cross-video accuracy (0.484) barely exceeds the majority baseline (0.470).**
   Only macro-F1 and kappa show information being carried. Two of four classes
   are effectively not learned.
3. **16 examples in the smallest class** (Delayed Support), spread over LOVO
   folds, is ~4 per fold. No learner recovers a class from that.
4. **The current CSVs predate the detector fix** — 3.97 players/frame against the
   5.8 the fixed pipeline now produces. Re-extraction is the first improvement.
5. **Distances are metric only with a homography.** Under the linear fallback
   they are relative, and should be reported as such.

---

## 5. Install and run

```bash
python -m venv .venv
```

```bash
pip install -r fyp/requirements.txt
```

Reproduce every metric and figure:

```bash
python reproduce.py
```

Analyse a video end-to-end:

```bash
python fyp/pipeline.py "match.mp4" tactical_ml.joblib --yolo-model yolo11n.pt --ball-model fyp/volleyball_best.pt --imgsz 1280 --team-split colour --tracker botsort --auto-court --court-coords linear --output-video annotated.mp4 --output-csv predictions.csv
```

Train the tactical classifier:

```bash
python fyp/ml_classifier.py training_csv --features invariant --save-model tactical_ml.joblib
```

Re-extract features after a detector change (**flags must match the pipeline**):

```bash
python fyp/prepare_training_data.py "dataset/dataset" --output-dir training_csv_v3 --yolo-model yolo11n.pt --ball-model fyp/volleyball_best.pt --imgsz 1280 --team-split colour --auto-court --court-coords linear --tracker botsort --clean-output
```

One-click runners: `MAKE_DEMO_VIDEO.bat`, `RETRAIN_AFTER_DETECTOR_FIX.bat`.
GPU retraining: `kaggle_retrain_after_detector_fix.ipynb`.

---

## 6. Breaking the label circularity (the next step)

Because the rule engine wrote the labels, no current score is a claim about
tactical accuracy. The protocol that converts it into one:

```bash
python fyp/annotate.py dataset/dataset --out gold_labels.csv --per-class 40
```

```bash
python fyp/ml_classifier.py training_csv --gold gold_labels.csv
```

`annotate.py` is deliberately **blind** (clips are shuffled across classes and
the rule label is never shown — otherwise the annotator is anchored to it and
the agreement becomes self-fulfilling) and **balanced** (equal clips per class,
because proportional sampling would give ~4 Delayed Support examples).

This then reports **rules-vs-human** and **model-vs-human** side by side. If the
model beats its own teacher against human judgement, it has genuinely denoised
the rules — the standard weak-supervision result (Ratner et al., *Snorkel*,
VLDB 2017). If it does not, that negative result is reported.

Ideally two annotators label the same clips, so human-vs-human kappa can be
quoted. **That is the real ceiling for every model here**, and no automated
score should be presented as exceeding it.

---

## 7. Repository map

| Path | Role |
|---|---|
| `reproduce.py` | one-command regeneration of every metric and figure |
| `fyp/pipeline.py` | end-to-end video → tactical report |
| `fyp/preflight.py` | pre-run population gate; aborts rather than emit a meaningless report |
| `fyp/interfaces.py` | detector + classifier abstractions, dual detector, backend factory |
| `fyp/features.py` | 18-dim `perm_invariant_v2` contract (single source of truth) |
| `fyp/teams.py` | jersey-colour team separation + per-track voting |
| `fyp/court.py` | homography, auto court detection, camera-motion compensation, mask |
| `fyp/tracking.py`, `fyp/identity.py` | tracker backends; persistent player slots |
| `fyp/ml_classifier.py` | learned tactical classifier, CV protocols, ablations |
| `fyp/annotate.py` | blind human annotation tool for the gold set |
| `fyp/mamba_model.py`, `fyp/train_mamba.py` | Mamba SSM comparison arm |
| `fyp/analytics.py`, `fyp/roles.py` | measured-vs-literature biomechanics, role inference |
| `fyp/tests/` | 144 tests, most runnable with no GPU |
| `Technical_Review_and_QA.md` | full audit log: every bug, its root cause, its fix |

---

## 8. Testing

```bash
python -m pytest fyp/tests -q
```

144 passed, 1 skipped. The suite encodes the project's own regressions: each
historical failure (§12.8 half-court homography, §12.9 slot starvation, §13.1
detector blindness, §13.7 crowd-captured colour clusters) has a test that
replays the real measured values. Every one of those failures ran to completion
and printed a formatted report — none raised an error. That is why the preflight
gate exists.

---

## 9. References

- Gu, A. & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* arXiv:2312.00752
- Zhang, Y. et al. (2022). *ByteTrack: Multi-Object Tracking by Associating Every Detection Box.* ECCV 2022
- Aharon, N. et al. (2022). *BoT-SORT: Robust Associations Multi-Pedestrian Tracking.* arXiv:2206.14651
- Ratner, A. et al. (2017). *Snorkel: Rapid Training Data Creation with Weak Supervision.* VLDB 2017
- Jocher, G. et al. *Ultralytics YOLO* (v8, v11). https://docs.ultralytics.com
- Sports-science sources for the biomechanical validation are listed in
  `Technical_Review_and_QA.md` §10.5.
