# Volleyball Tactical Analysis with Mamba SSM
## Technical Review, Correctness Audit, and Q&A for a Computer-Vision Expert

**Prepared for:** review by a computer-vision specialist with volleyball domain knowledge
**Scope:** YOLO detection → ByteTrack → Mamba SSM tactical classifier (4 classes)
**Status of project:** ~50% complete; this document audits correctness, fixes the
critical loopholes, answers the outstanding technical questions, and grounds the
tactical model in real sports-science data.

---

## 0. How to read this document

The project pipeline is sound in *shape* (detect → track → project to court →
window → sequence-classify). The problems are in the *data and the feature
contract*, and those problems silently corrupt results without throwing errors.
Section 1 is the evidence (measured on your own 285 clips), Section 2 is what was
fixed in code, Sections 3–6 answer every question you raised, and Section 7 is
the roadmap to a defensible final submission.

Everything in Section 1 was measured by `fyp/diagnose_data.py` on your real data,
and every fix in Section 2 is covered by a passing unit test in
`fyp/tests/test_features.py` (9/9 passing).

---

## 1. Data-health audit (measured on your 285 clips)

| Finding | Measured value | Why it matters |
|---|---|---|
| **Class imbalance** | Coordinated Defense **134**, Coordinated Attack **90**, Spacing Breakdown **45**, Delayed Support **16** | Delayed Support has 16 examples for a 4-class deep model. It cannot be learned; it drags macro-F1 down. This is the single biggest reason val macro-F1 sits at ~0.52. |
| **Ball channel divergence** | `ball_x` range **−14,105 … +5,726**, `ball_y` up to **+11,671** | The constant-velocity ball estimator extrapolates off to infinity when detections are missing. On a 1800×900 cm court these are impossible. Because the model z-scores features, a few ±14,000 outliers dominate the ball statistics and inject noise. |
| **Ball missing rate** | **42%** of frames per clip (mean) | Two of the four hand-coded rules (Coordinated Attack, Delayed Support) depend on the ball. With the ball absent/garbage 42% of the time, those *teacher labels* are themselves unreliable. |
| **Coordinate-convention mismatch** | **31 clips** contain `ball_y < 0` | The training data uses a "net at Y=0, other side negative" convention, while `pipeline.py` linearly maps pixels into a 0–1800 × 0–900 court with the net at Y=450 — which can never be negative. Training distribution ≠ inference distribution. |
| **Overfitting** | train_acc → **1.00**, val_acc ~0.66, val **macro-F1 ~0.52** | 4-class chance is 0.25. The model memorises 285 noisy clips and generalises weakly. Classic small-data + label-noise + imbalance signature. |
| **Identity stability** | median slot-`p1` motion **4.2 cm/frame** (sane), but individual clips show **>900 cm** single-frame jumps | Player columns `p1..p6` are *not* guaranteed to track the same physical player. Most frames are fine, but slot swaps occur (verified in `Coordinated_Attack…clip_002`, frames 1→3), which spikes any per-slot velocity feature. |

**Headline:** the architecture is not the problem — the *labels and features* are.
The Mamba model is being asked to imitate a noisy rule-based teacher on a small,
imbalanced, partially-corrupted dataset. Fix the data contract first.

---

## 2. Correctness loopholes fixed in code

All fixes live in a new, dependency-light module `fyp/features.py` (pure NumPy +
optional SciPy) and are wired into `pipeline.py`, `train_mamba.py` and
`infer_mamba.py` so that **training, inference and the live video pipeline build
byte-for-byte identical model inputs** (train/serve parity). Each fix has a test.

### L1 — Player identity swap → corrupted velocity
**Problem:** `pipeline._extract_positions` re-sorted tracks by "age" every frame
and packed them into slots 0–5, so the player in a given slot kept changing;
`np.diff` over a slot then measured the jump between *different people*.
**Fix:**
- `pipeline.py` now keeps a persistent `slot_map: {tracker_id → slot}`; once a
  ByteTrack ID owns a slot it keeps it for the clip.
- `features.recover_identity()` re-threads the existing (already-swapped) CSVs by
  Hungarian assignment (greedy fallback without SciPy), so historical data is
  repaired on load.
- All kinematic features (`sync_score`, centroid velocity, nearest-neighbour
  spacing) are recomputed on identity-consistent tracks.

### L2 — `(0,0)` "missing" sentinel
**Problem:** untracked players/ball were written as `(0,0)`, a *real* court
corner. The model cannot tell "at the corner" from "not seen", and the
transition real → (0,0) → real injects a fake velocity spike.
**Fix:** missing = **NaN** with an explicit presence flag; `(0,0)` is converted
to NaN on load; features are NaN-aware.

### L3 — Permutation sensitivity
**Problem:** a formation is a *set* of players; listing order should not change
the prediction, but raw `p1..p6` coordinates make order meaningful.
**Fix:** the model now consumes a **permutation-invariant per-frame descriptor**
(`features.team_features`): ball presence + position, team centroid, spread,
convex-hull area, nearest-neighbour distance stats, max pair distance, players
present, ball-to-centroid distance — 14 features, all order-independent. The
input dimension stays 14, so the Mamba architecture is unchanged.

### L4 — Divergent ball extrapolation
**Problem:** ball coordinates blew up to ±14,000 cm.
**Fix:** `features._sanitize` rejects any coordinate outside a generous court
envelope (±900 cm margin) as missing — verified by `test_divergent_ball_rejected`.

### L5 — Distance bias from box centroid
**Problem:** player court position used the bounding-box **centroid**, which
floats ~half a body up-court and biases distance with camera depth.
**Fix:** `features.foot_point` uses the **bottom-centre** of the box (the feet),
the only point that lies on the floor plane.

### L6 — Stale-checkpoint guard
Because the input semantics changed, the old checkpoint
(`mamba_checkpoint_team_bottom.pt`) is now invalid. Checkpoints are tagged
`feature_version="perm_invariant_v1"`; `pipeline.py`/`infer_mamba.py` print a loud
warning if an old checkpoint is loaded, so you can never silently serve garbage.
**Action required: retrain** (`python train_mamba.py training_csv --augment`).

### L7 — Overfitting mitigation
`train_mamba.py` gains label-preserving augmentation (`--augment`): Gaussian
position jitter (sensor-noise model) + random horizontal court mirror (valid for
all four classes). Combined with the existing inverse-frequency class weights,
label smoothing, weight decay, cosine LR and macro-F1 early-stopping, this is the
right regulariser set for ~285 samples.

### New diagnostics & tests
- `fyp/diagnose_data.py <dir>` — prints the Section 1 table for any clip folder.
- `fyp/tests/test_features.py` — 9 unit tests, all passing, runnable with no GPU:
  `python fyp/tests/test_features.py`.

---

## 3. Computer-vision questions answered

### 3.1 How is distance calculated?
Two stages. **(a) Pixel → court projection.** Each player's image point is mapped
to court centimetres either by a linear pixel scale (default) or, correctly, by a
**homography** `H` (Section 3.3). The mapped point is `(x_cm, y_cm)`.
**(b) Distance.** All distances are plain Euclidean in court space:
`d = sqrt((x_i − x_j)² + (y_i − y_j)²)`, computed pairwise with NumPy broadcasting
in `features._pairwise_dist` and in the rules. Two corrections were applied:
- the player point is now the **feet** (bottom-centre of box), not the centroid,
  so the projected point actually lies on the floor (L5);
- distances skip **NaN** (missing) players, so an absent player no longer creates
  a phantom 0-cm or huge-cm pair (L2).
Court distance is only metric-accurate **with** a homography; under the linear
fallback it is an affine approximation and should be reported as "relative", not
absolute cm.

### 3.2 Corner calibration — frame of reference, and how corner values are obtained
The court is a known real-world rectangle: **18 m × 9 m** (full court) — or you can
calibrate to one half (9 m × 9 m) or to specific lines (e.g. 3 m attack line).
That known geometry **is** the frame of reference: you define court coordinates in
cm with a chosen origin (e.g. a corner or the net-centre) and axes (X along the
sideline, Y along the endline).

How the four corner pixel values are obtained, in order of rigour:
1. **Manual click (recommended for a fixed broadcast camera):** open one frame,
   click the four court corners (or any four points whose court coordinates you
   know — antenna bases, 3 m line intersections), read off the pixel `(u,v)`.
   These are the `--court-corners TL TR BR BL` you pass to `pipeline.py`.
2. **Line detection (semi-automatic):** detect court lines (Hough transform or a
   learned court-segmentation net), intersect them to get corners.
3. **Calibration target:** if you control capture, place markers at known court
   points.
Because four point correspondences (pixel ↔ known court-cm) fully determine a
planar homography, four corners are the minimum; using 6–8 points and a
least-squares fit (`cv2.findHomography` with RANSAC) is more robust to click
error. **Key caveat:** the homography is valid only for the floor plane and only
while the camera does not move/zoom — re-calibrate per camera setup.

### 3.3 Perspective transformation — what it is and why you need it
A camera maps the 3-D world to 2-D pixels by a projective transform. For points on
a single plane (the court floor) the pixel↔court mapping is exactly a **homography**
`H` (a 3×3 matrix, 8 DOF) acting on homogeneous coordinates:
```
[u', v', w']ᵀ = H · [X, Y, 1]ᵀ ,   (x_court, y_court) = (u'/w', v'/w')
```
`cv2.findHomography(src_pixels, dst_court_cm)` estimates `H`;
`cv2.perspectiveTransform` applies it (both already used in `pipeline._build_homography`).
Why it matters: **without** it, equal pixel distances near and far from the camera
correspond to very different real distances, so "spacing" and "velocity" are
distorted by depth. All four tactical classes are defined by spacing/velocity, so
the homography is not optional polish — it is what makes the features mean what the
labels claim. The linear fallback should be treated as a debug mode only.

### 3.4 YOLOv8 → YOLOv11: is the switch justified?
Short answer: **yes for detection quality, but it is not your bottleneck — and the
bigger win is switching to a pose model, not a newer detector.**
- **v11 vs v8:** YOLOv11 (Ultralytics, 2024) gives modestly better mAP at similar
  or lower latency and is a drop-in (`YOLO("yolo11n.pt")` / `yolo11s.pt`). It is a
  reasonable, cheap upgrade. It does **not** fix your real issues (labels, ball,
  imbalance), so justify it as "current SOTA detector, marginal accuracy gain,
  zero integration cost", not as a solution to model performance.
- **Generic COCO weights are the actual detection weakness:** the volleyball is a
  small, fast COCO "sports ball" — hence your 42% miss rate. Two better options:
  (i) fine-tune detection on volleyball footage, or (ii) use a dedicated
  small-fast-object tracker for the ball.
- **Strong recommendation:** move to **YOLOv11-Pose** (or v8-Pose). It returns 17
  COCO keypoints per player, which directly enables the limb features you asked
  about in 3.5 and gives a far better court-contact point (ankles) than any box
  heuristic. That single change improves L5 (distance), enables richer tactics,
  and modernises the detector in one step.
Recommended phrasing for the expert: *"We benchmark YOLOv8n vs YOLOv11n/s on a
held-out annotated set (mAP@50 and ball recall) and adopt YOLOv11-Pose for the
keypoint features; the detector is not the performance bottleneck, the teacher
labels are."*

### 3.5 Adding player features (hand, elbow, knee, …) for better identification
This is the highest-value capability upgrade, and it pairs with 3.4. With a **pose
estimator** you get per-player keypoints (shoulders, elbows, wrists, hips, knees,
ankles). From them you can derive features the current centroid model is blind to:
- **Arm elevation / wrist-above-head:** a near-binary spike/block detector — the
  hands going above the head is the visual signature of an attack or block, far
  more discriminative than "two players moving fast".
- **Knee/hip flexion → load-and-jump:** detects the crouch that precedes a jump
  (anticipation), enabling earlier, cleaner tactical segmentation.
- **Body orientation (shoulder-hip vector):** who is facing the ball — directly
  encodes "is this player engaged or out of the play" (relevant to Delayed
  Support and Spacing Breakdown).
- **Ankle midpoint** as the court-contact point: best possible foot location.
How to integrate without breaking the design: keep the permutation-invariant
contract from L3 — aggregate per-player keypoint features into **team-level,
order-independent** statistics (e.g. "fraction of players with wrists above head",
"max knee-flexion in the group", "mean body-orientation alignment to ball") and
append them to `features.PERM_INVARIANT_COLS`. That widens `INPUT_DIM` and requires
a retrain, but it is the cleanest path to a model that recognises volleyball
actions rather than just blobs moving.

### 3.6 Why is there "no intelligence in tactical deviations"?
Because of a circularity you have correctly sensed: the labels come from
`label_clips.py` (four hand-coded thresholds), and the Mamba model is trained to
**imitate those rules**. By construction it cannot detect anything the rules don't
already encode — it can at best approximate them, and it inherits their noise. It
has no notion of "this play deviates from the team's normal pattern" because it was
never shown deviations as a concept.
Three ways to add real deviation intelligence:
1. **Anomaly / novelty detection (unsupervised):** train an autoencoder or a
   one-class model on "normal" coordinated sequences; high reconstruction error =
   tactical deviation. This needs *no* labels and directly answers the question.
2. **Self-supervised pretraining:** train the Mamba to predict the next frame's
   team descriptor (forecasting), then a deviation is a frame where predicted ≠
   observed by a large margin. This also gives you a better initialisation that
   reduces overfitting.
3. **Human-validated labels for the deviation classes** (Delayed Support, Spacing
   Breakdown) so the model learns deviations from ground truth rather than from a
   thresholded teacher. Even 150–200 expert-labelled clips would transform the
   minority classes.
Recommended: ship (1) as a "tactical-deviation score" alongside the 4-class output
— it is a strong, honest, novel contribution and sidesteps the teacher-noise trap.

### 3.7 The Mamba model — what it is and is it the right choice
**What it is:** Mamba (Gu & Dao, 2023) is a *selective state-space model*. A linear
state-space layer maintains a hidden state `h_t = Ā h_{t−1} + B̄ x_t`,
`y_t = C h_t`; "selective" means `B`, `C`, and the timestep `Δ` are
**input-dependent**, so the model learns *what to remember and what to forget* per
token. It runs in linear time in sequence length (vs a Transformer's quadratic
attention) and your `mamba_model.py` implements the S6 recurrence in pure PyTorch
(no CUDA kernel needed) — correct and readable.
**Is it justified for this task?** Honest assessment:
- For 29-frame sequences, sequence length is tiny, so Mamba's headline advantage
  (linear-time long-context) is **not** the reason to use it here. An LSTM, a 1-D
  CNN, or a small Transformer would all be reasonable and would train faster on
  285 samples.
- The defensible justifications are: (i) Mamba is a strong, modern sequence model
  with good inductive bias for continuous-time trajectories; (ii) it scales
  cleanly if you later classify **longer** windows (whole rallies), where it
  genuinely beats Transformers; (iii) it is a novel, publishable choice for sports
  tactics. Frame it as "chosen for trajectory modelling and future long-rally
  scalability", not "needed for 29 frames".
- **Caution to state up front:** with 285 samples a 4-layer, 64-dim Mamba
  (~hundreds of k params) overfits (Section 1). Report parameter count vs sample
  count and show the augmentation/regularisation that addresses it; consider
  `--d_model 32 --n_layers 2` as a smaller-capacity baseline for comparison.

### 3.8 Why 29 frames?
29 frames at 30 FPS ≈ **0.97 s** — about the duration of one tactical beat
(approach + jump + contact, or a defensive shift). It is a reasonable window, but
the real reason in your code is mechanical: each raw clip is 41 frames, split into
a **29-frame training window** (frames 1–29) and a **12-frame outcome window**
(frames 30–41) used by `label_clips.py` to derive the label. So "29" = "41 minus
the 12-frame look-ahead the labeller needs". That is a fine design, but state it
explicitly and justify both numbers:
- **Window ≈ 1 s** captures a single action without blending two rallies.
- **12-frame (0.4 s) outcome horizon** is enough to see the consequence (did the
  ball get attacked / did support arrive late).
If you want to defend it rigorously, run a small ablation over window lengths
{15, 23, 29, 45} and report macro-F1 — that turns "why 29" from an assertion into
evidence. Note the model also accepts other lengths; only the data-prep constants
(`EVAL_START`, `TRAIN_END`) encode 29/41.

### 3.9 The "Unclassified" class
Current behaviour: in `label_clips.py`, clips where no rule fires are labelled
**"Unclassified" and deleted**; in `pipeline.py` a live sequence that no rule
matches is forced to **"Spacing Breakdown"**. Both are problematic — deleting
discards hard cases and biases the dataset toward easy, rule-friendly clips, and
the forced default fabricates a label.
Recommendations (pick per goal):
- **Add an explicit 5th "Unclassified / Transition" class** and *keep* those clips.
  Most real video is between-rally or ambiguous; a model that can say "no clear
  tactic" is more honest and more useful. This requires `NUM_CLASSES=5` and a
  retrain.
- **Or add confidence-based abstention at inference:** if max softmax < τ (e.g.
  0.5), output "Uncertain" instead of forcing a class. Cheap, no retrain, and it
  visibly improves trust in the live overlay.
Either way, stop deleting Unclassified clips — that deletion is part of why the
dataset looks cleaner than the real task is.

---

## 4. From individual skill to team tactics: a data-grounded justification

You asked to ground the tactical model in real player data — that an international
attacker is explosively different from a club player, and that each role
(attacker, setter, middle blocker, libero) has a measurable physical signature
which, aggregated, produces the team behaviours your four classes describe. This
section provides that, with sourced figures, and maps each biometric to a
**CV-observable proxy** the pipeline can actually measure.

### 4.1 One correction up front (so the expert trusts your numbers)
A common mix-up: elite players do **not** "jump 2 m". The **vertical jump** of an
elite spiker is ~**0.55–0.90 m**; what reaches ~**3.5 m** is the **spike reach**
(standing reach + jump + arm extension above a 2.43 m men's net). Quote *spike
reach ≈ 350 cm* and *vertical jump ≈ 70–90 cm*, not "2 m jump". Using the right
quantity is itself a credibility signal.

### 4.2 Elite vs developmental — the gap is real and measurable
- **Spike (attack) reach:** elite men avg **354.5 cm**, elite women **309.4 cm**;
  college-level women average ~**280 cm**. The elite-vs-club gap is ~30–70 cm of
  reach — i.e. the elite attacker contacts the ball measurably higher and earlier.
- **Attack jump height:** elite attackers ~**64 cm** (sport-specific approach
  jump), with higher jumps correlating with game efficiency.
- **Approach run-up:** elite attackers start ~**4–4.5 m** from the net and convert
  horizontal run-up into a vertical take-off velocity of ~**2.9 m/s** — the
  "explosive" quality you described, now as a number a tracker can estimate from
  centroid speed in the last ~5 frames before take-off.

### 4.3 Positional biometric signatures (sourced)

| Role | Height (typical) | Spike / block reach | Jump (CMJ) | Distinctive physical trait | CV-observable proxy in your pipeline |
|---|---|---|---|---|---|
| **Middle blocker** | tallest (~**194 cm**), ~85 kg | **highest** reach (with opposite) | **highest CMJ ~36 cm** | tall, explosive, lateral net mobility | tallest bounding boxes; fastest lateral (X) centroid shift along the net; arms-up keypoints |
| **Outside hitter** | shortest/lightest of the hitters | high reach, below MB/OPP | high | mobile (also receives serve) | high approach velocity from the wing; wide court start position |
| **Opposite (RS)** | tall, heavy (~MB) | reach ≈ MB | high | right-side power attacker | high wrist-above-head rate on the right zone |
| **Setter** | mid height | **lower** reach | moderate CMJ, **highest braking force & RFD** | quick, low, high rate-of-force-development for fast repositioning | short, high-frequency moves to the ball; central hub of nearest-neighbour graph |
| **Libero** | **shortest, lightest** | lowest (never blocks/attacks at net) | lowest | fastest reaction/defensive reflex; back-court only | always back-row (high Y); never crosses to net zone; fastest dig reactions |

Sources for the table: positional anthropometry and CMJ from IJERPH 2021 and the
PMC reviews below; setter braking/RFD from the Division-I force-velocity study;
reach ordering (MB ≈ OPP > OH > S/L) from the professional positional-differences
study.

### 4.4 Reaction time — the libero/defence axis (relevant to "Delayed Support")
Defensive reaction is measurable: studies report whole-body/upper-extremity
reaction times in the **~650–770 ms** range (improving with training), with middle
and back-court defenders showing faster choice-reaction at high stimulus speeds.
This is the literature anchor for your **Delayed Support** class: a "late" support
player is one whose movement onset after ball contact exceeds the normal reaction
window. Your rule encodes exactly this (peak-speed frame > 5 frames after impact ≈
>0.17 s lag on top of baseline) — now you can cite *why* that threshold is
physiologically reasonable rather than arbitrary.

### 4.5 How individual signatures aggregate into your four team classes
- **Coordinated Attack** = 1–2 players showing the *attacker signature*
  (high approach velocity, wrist-above-head) while base players (setter after the
  set, off-blockers) stay low-velocity near the net → your "top-2 fast, bottom-4
  slow" rule is the team-level shadow of individual attacker biomechanics.
- **Coordinated Defense** = the whole unit translating together (the *block-shift*
  middle-blocker lateral signature propagating to the line) with low internal
  spread variance → "centroid moving, formation preserved".
- **Delayed Support** = the nearest defender's *reaction-time* signature exceeding
  the normal dig window after impact.
- **Spacing Breakdown** = the role spacing that normally tiles the court (libero
  deep, MB at net, OH on the wing) collapsing or over-stretching → hull
  area/nearest-neighbour outliers.
This is the justification chain the expert will want: **biomechanics → individual
CV proxy → permutation-invariant team feature → tactical label.**

### 4.6 The Russell–Lange Volleyball Test — where it fits
The **Russell–Lange Volleyball Test** is a classic standardized skills battery
(serving and repeated volleying/passing accuracy) used in physical education to
quantify *individual* volleyball skill off the ball of play. It is worth citing as
the conceptual ancestor of what you are doing: where Russell–Lange measures skill
in isolated drills, your system measures skill expression **in live game context**
(positioning, timing, coordination) automatically from video. Framing your work as
"an automated, in-situ successor to standardized skill tests" is a clean way to
motivate the project for a sports-science audience. (Note: Russell–Lange assesses
isolated skill, not team coordination, so use it to motivate, not to validate, the
tactical labels.)

### 4.7 Research sources to cite (PubMed / journals)
These are the primary sources behind Section 4 and good citations for your report.
For motion-capture/biomechanics methodology (the "sport motion analysis" and
"biometric parameter" threads), the spike-jump kinematics and force-velocity
papers are the most directly relevant.

---

## 5. Roadmap to a defensible, "foolproof" submission

Ordered by impact-per-effort:

1. **Retrain on the corrected pipeline (required).** The old checkpoint is invalid.
   `python train_mamba.py training_csv --augment --epochs 80 --val_split 0.2 --test_split 0.1`
   Report held-out **test** macro-F1 and the confusion matrix, not train accuracy.
2. **Fix the label source for minority classes.** Hand-label ~150–200 clips
   (especially Delayed Support, n=16, and Spacing Breakdown) so those classes have
   real ground truth. This will move macro-F1 more than any model change.
3. **Calibrate the homography per camera** (Section 3.2) and re-extract court
   coordinates, so training and inference share one metric court frame. This also
   removes the negative-`ball_y` convention mismatch.
4. **Replace the ball estimator.** Cap extrapolation (already sanitised), and
   prefer a dedicated small-object/ball tracker or a fine-tuned ball class; treat
   ball-derived rules as low-confidence when ball recall is low.
5. **Move to YOLOv11-Pose** and add the limb/orientation team features (Sections
   3.4–3.5). Widen `INPUT_DIM`, retrain.
6. **Add a tactical-deviation score** via unsupervised anomaly detection
   (Section 3.6) — your novel contribution, and it needs no labels.
7. **Add the Unclassified handling** (5th class or abstention, Section 3.9); stop
   deleting hard clips.
8. **Run the ablations** the expert will ask for: window length (3.8), model size
   (3.7), v8 vs v11 detector + ball recall (3.4). Evidence beats assertion.

### Honest limitations to state in the report (credibility, not weakness)
- Labels are from a rule-based teacher; the model imitates rules and inherits
  their noise (mitigated, not eliminated).
- 285 clips with severe imbalance (16 in the smallest class) → small-data regime;
  results are reported with held-out test + augmentation, but more/labelled data
  is the real fix.
- Court metricity depends on a per-camera homography; without it, distances are
  relative, not absolute cm.

### What is already fixed and verified in this pass
- Identity-stable, NaN-aware, permutation-invariant, divergence-robust features
  (`fyp/features.py`), wired identically into train/infer/live pipeline.
- Foot-point court contact; checkpoint version guard; augmentation flag.
- 9/9 unit tests passing (`python fyp/tests/test_features.py`); data audit tool
  (`python fyp/diagnose_data.py training_csv`).

---

## 6. Quick commands

```bash
# 0. install (adds scipy for optimal identity matching)
pip install -r fyp/requirements.txt

# 1. audit your data (prints the Section 1 table)
python fyp/diagnose_data.py training_csv

# 2. run the correctness unit tests (no GPU needed)
python fyp/tests/test_features.py

# 3. RETRAIN (old checkpoint is invalid under the new feature contract)
python train_mamba.py training_csv --augment --epochs 80 \
    --val_split 0.2 --test_split 0.1 --checkpoint mamba_checkpoint_v2.pt

# 4. inference on clip CSVs
python infer_mamba.py mamba_checkpoint_v2.pt training_csv --output_csv preds.csv

# 5. full video pipeline (supply real court corners for metric distances)
python pipeline.py match.mp4 mamba_checkpoint_v2.pt \
    --court-corners 42,18 1238,18 1238,702 42,702 \
    --output-video annotated.mp4 --output-csv seq_preds.csv
```

---

## 7. References

**Models / methods**
- Gu, A. & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* arXiv:2312.00752.
- Zhang, Y. et al. (2022). *ByteTrack: Multi-Object Tracking by Associating Every Detection Box.* ECCV 2022.
- Jocher, G. et al. *Ultralytics YOLO* (v8, v11). https://docs.ultralytics.com

**Volleyball biomechanics & positional science (PubMed / journals)**
- Influence of jump height on game efficiency in elite volleyball players. *Scientific Reports* (2023). https://pmc.ncbi.nlm.nih.gov/articles/PMC10235019/
- Spike jump biomechanics, male vs female elite players. *J. Sports Sciences* (2019). https://www.tandfonline.com/doi/full/10.1080/02640414.2019.1639437
- Anthropometric & vertical-jump abilities by position/level (junior female). *IJERPH* (2021). https://pmc.ncbi.nlm.nih.gov/articles/PMC8393901/
- Anthropometric, physical, and age differences by position and level in volleyball. *PMC*. https://pmc.ncbi.nlm.nih.gov/articles/PMC4327374/
- Positional jump loads & force–velocity metrics over a D-I season (setter braking/RFD). *PMC*. https://pmc.ncbi.nlm.nih.gov/articles/PMC11669427/
- Kinematic analysis of the volleyball attack / approach take-off velocity. *J. Human Kinetics* (2017). https://pmc.ncbi.nlm.nih.gov/articles/PMC5548173/
- Reaction time and playing position in volleyball. *The Sport Journal*. https://thesportjournal.org/article/comparison-of-coinciding-anticipation-timing-and-reaction-time-performances-of-adolescent-female-volleyball-players-in-different-playing-positions/

**Standardized skill assessment**
- Russell, F. & Lange, E. *Russell–Lange Volleyball Test* (serving & repeated volleying) — classic PE skills battery; see standard tests-and-measurements texts in physical education.

---

*Prepared as a correctness audit + technical Q&A. All Section 1 numbers were
measured on the project's own 285 clips; all Section 2 fixes are covered by passing
unit tests.*
