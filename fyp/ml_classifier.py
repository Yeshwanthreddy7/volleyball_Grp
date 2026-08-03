"""
ml_classifier.py - learned tactical classifier (scikit-learn), replacing the
hand-coded threshold engine as the DECISION mechanism.

READ THIS BEFORE PRESENTING ANY NUMBER FROM THIS FILE
-----------------------------------------------------
The four class labels in `training_csv/*.csv` were produced by
`label_clips.py`, a rule engine with hand-tuned thresholds. Any supervised
model trained on those labels is, strictly, a function approximator OF THOSE
RULES. Swapping "if spacing > 340 cm" for "RandomForest" and reporting 90%
agreement proves only that a forest can memorise a threshold - a computer-vision
examiner will spot the circularity in one question, and a volleyball expert will
ask the harder version: "does the model agree with a COACH, or with your rules?"

This module is therefore built to make the non-circular claim measurable, in
three explicit tiers:

  TIER 1 (available today, no new labels)
      Learned decision boundary over the full 29-frame sequence instead of
      per-frame thresholds. Legitimately better than the rule engine because
      the rules read a handful of scalars at fixed cut-points, while the model
      reads the temporal SHAPE of 18 channels and learns the cut-points from
      data. Report as "agreement with the heuristic teacher", never as
      "tactical accuracy".

  TIER 2 (needs the gold set from `fyp/annotate.py`, ~2 h of human labelling)
      Train on rule labels, TEST on human labels. If the model scores higher
      against humans than the rules do, it has genuinely denoised its teacher -
      the standard, published result of weak supervision (Ratner et al.,
      Snorkel, VLDB 2017). THIS is the number that answers the examiner.

  TIER 3 (gold set as training data)
      Fine-tune or retrain on human labels; rules become one weak signal among
      several. This is the honest endpoint.

`--gold` runs Tier 2 automatically when a gold CSV is supplied.

WHY CLASSICAL ML AND NOT ONLY THE MAMBA SSM
-------------------------------------------
With 285-508 clips and 16 examples in the smallest class, a 4-layer selective
state-space model is heavily over-parameterised - the project already measured
train_acc 1.00 vs val_acc 0.66. Gradient-boosted trees and SVMs are the correct
model class for this sample size, they train in seconds (so leave-one-video-out
is cheap), and they expose permutation importance, which turns "the model said
Coordinated Attack" into "because top-2 player speed rose while spacing
collapsed" - the only form a volleyball expert can actually argue with. The
Mamba model stays in the project as the deep-learning comparison arm; this
module is the evidence that a simpler learner is not being ignored.

Usage
-----
    python fyp/ml_classifier.py training_csv --output-dir ml_results
    python fyp/ml_classifier.py training_csv --gold gold_labels.csv
    python fyp/ml_classifier.py training_csv --save-model tactical_rf.joblib
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from features import (  # noqa: E402
    MODEL_FEATURE_COLS,
    build_model_sequence,
)
from train_utils import video_key  # noqa: E402

SEQ_LEN = 29
CLASS_NAMES = [
    "Coordinated Attack",
    "Coordinated Defense",
    "Delayed Support",
    "Spacing Breakdown",
]


# ---------------------------------------------------------------------------
# 1. Sequence -> fixed-length feature vector
# ---------------------------------------------------------------------------

# Each of the 18 permutation-invariant channels is summarised by these six
# statistics. The choice is deliberate rather than exhaustive:
#   mean/std   - level and variability (a formation's spread and its steadiness)
#   min/max    - the extremes that define an event (peak speed IS the spike)
#   slope      - direction of travel over the window; this is the channel the
#                rule engine cannot see at all, because it reads scalars, not
#                trends. "Spacing widening" and "spacing already wide" are
#                different tactical situations and only slope separates them.
#   iqr        - robust spread, unaffected by a single tracking glitch
AGGREGATIONS = ("mean", "std", "min", "max", "slope", "iqr")


def _slope(x: np.ndarray) -> float:
    """Least-squares slope per frame, NaN-aware. 0.0 when under-determined."""
    t = np.arange(len(x), dtype=float)
    m = np.isfinite(x)
    if m.sum() < 2:
        return 0.0
    t, y = t[m], x[m]
    t = t - t.mean()
    denom = float((t * t).sum())
    if denom <= 0:
        return 0.0
    return float((t * (y - y.mean())).sum() / denom)


def sequence_features(seq_raw: np.ndarray) -> np.ndarray:
    """
    (29, 14) raw clip rows -> (18 * 6,) feature vector.

    Routes through `features.build_model_sequence`, so this shares ONE feature
    contract with the Mamba model, `pipeline.py` and `infer_mamba.py`. A model
    trained here can therefore be served live without a second feature path -
    the train/serve mismatch class that cost this project two debugging cycles
    (L6, sec 12.8, sec 13.7).
    """
    return absolute_from_model_sequence(
        build_model_sequence(np.asarray(seq_raw, dtype=float),
                             target_len=SEQ_LEN))


def absolute_from_model_sequence(seq: np.ndarray) -> np.ndarray:
    """Aggregate an already-built (29, 18) model sequence.

    Split out from `sequence_features` because `pipeline.py` hands the
    classifier the MODEL sequence, not the raw 14-column rows - so training and
    live serving must share this half of the path, not just the earlier half.
    """
    out = []
    with warnings.catch_warnings():
        # All-NaN channels are expected (a clip where the ball was never seen);
        # they become NaN here and are imputed inside the sklearn Pipeline.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for c in range(seq.shape[1]):
            ch = seq[:, c]
            out.extend([
                np.nanmean(ch),
                np.nanstd(ch),
                np.nanmin(ch),
                np.nanmax(ch),
                _slope(ch),
                np.nanpercentile(ch, 75) - np.nanpercentile(ch, 25),
            ])
    return np.asarray(out, dtype=float)


def feature_names() -> list[str]:
    return [f"{col}__{agg}" for col in MODEL_FEATURE_COLS for agg in AGGREGATIONS]


# ---------------------------------------------------------------------------
# 1b. Scale-invariant features - the fix for cross-video collapse
# ---------------------------------------------------------------------------
#
# WHY THIS EXISTS
# ---------------
# Measured on the 285-clip set, the absolute feature block scores k-fold
# macro-F1 0.52 but leave-one-video-out macro-F1 0.27 with Cohen's kappa 0.08 -
# i.e. essentially no agreement beyond chance once the camera changes.
#
# The cause is dimensional, not statistical. Without a per-camera homography the
# pipeline falls back to a LINEAR pixel->court scaling, so "cm" means something
# different in every video: a 300 cm nearest-neighbour distance in
# videoplayback (1) and in videoplayback (4) are different physical distances.
# Channels like centroid_x, hull_area, max_pair_dist and ball_x are therefore
# camera-identity features. A learner given them will happily use them to
# recognise WHICH VIDEO it is looking at - exactly the per-video signature
# reported in sec 12.6 - and that skill transfers to no new match.
#
# The fix is to feed the model only quantities that survive an unknown scale
# factor:
#   * RATIOS of two lengths (dimensionless by construction), and
#   * the SHAPE of each channel over time, via per-clip z-scoring, which removes
#     that clip's own level and scale and keeps only how the signal moved.
#
# This is also the tactically correct representation. "The formation is 340 cm
# wide" is a camera measurement; "the formation is twice as wide as its tightest
# pair, and widening" is a volleyball observation.
_EPS = 1e-6
_CH = {c: i for i, c in enumerate(MODEL_FEATURE_COLS)}

# (name, numerator channel, denominator channel) - each is a length/length or
# speed/speed ratio, so any per-camera scale factor cancels.
RATIO_FEATURES = [
    ("compactness",      "nn_dist_mean",     "max_pair_dist"),
    ("tightest_pair",    "nn_dist_min",      "nn_dist_mean"),
    ("loosest_pair",     "nn_dist_max",      "nn_dist_mean"),
    ("spread_aspect",    "spread_x",         "spread_y"),
    ("ball_offset",      "ball_to_centroid", "max_pair_dist"),
    ("speed_ratio",      "speed_top2",       "speed_bot4"),
    ("speed_diff_share", "speed_diff",       "speed_top2"),
]

# Already dimensionless: a count, a cosine similarity and a 0/1 flag.
DIMENSIONLESS_CHANNELS = ("n_present", "sync_inst", "ball_present")

# Aggregations applied to z-scored channels. mean and std are omitted because
# z-scoring fixes them at 0 and 1 - keeping them would add two constant columns.
SHAPE_AGGREGATIONS = ("min", "max", "slope", "iqr")


def _zscore(x: np.ndarray) -> np.ndarray:
    m = np.isfinite(x)
    if m.sum() < 2:
        return np.zeros_like(x)
    mu = float(np.nanmean(x))
    sd = float(np.nanstd(x))
    if sd < _EPS:
        return np.zeros_like(x)
    return (x - mu) / sd


def _agg(ch: np.ndarray, which) -> list[float]:
    fns = {
        "mean": lambda a: np.nanmean(a),
        "std": lambda a: np.nanstd(a),
        "min": lambda a: np.nanmin(a),
        "max": lambda a: np.nanmax(a),
        "slope": _slope,
        "iqr": lambda a: np.nanpercentile(a, 75) - np.nanpercentile(a, 25),
    }
    return [float(fns[w](ch)) for w in which]


def invariant_sequence_features(seq_raw: np.ndarray) -> np.ndarray:
    """(29, 14) raw rows -> scale-invariant feature vector.

    Three blocks: dimensionless ratios, already-dimensionless channels, and the
    temporal shape of every channel after per-clip z-scoring.
    """
    return invariant_from_model_sequence(
        build_model_sequence(np.asarray(seq_raw, dtype=float),
                             target_len=SEQ_LEN))


def invariant_from_model_sequence(seq: np.ndarray) -> np.ndarray:
    """Scale-invariant aggregation of an already-built (29, 18) sequence."""
    out: list[float] = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        for _name, num, den in RATIO_FEATURES:
            a = seq[:, _CH[num]]
            b = seq[:, _CH[den]]
            r = a / (np.abs(b) + _EPS)
            r = np.clip(r, -50.0, 50.0)     # a near-zero denominator must not
            out.extend(_agg(r, AGGREGATIONS))   # become an outlier that dominates

        for col in DIMENSIONLESS_CHANNELS:
            out.extend(_agg(seq[:, _CH[col]], AGGREGATIONS))

        for c in range(seq.shape[1]):
            out.extend(_agg(_zscore(seq[:, c]), SHAPE_AGGREGATIONS))

    return np.asarray(out, dtype=float)


def invariant_feature_names() -> list[str]:
    names = [f"{n}__{a}" for (n, _, _) in RATIO_FEATURES for a in AGGREGATIONS]
    names += [f"{c}__{a}" for c in DIMENSIONLESS_CHANNELS for a in AGGREGATIONS]
    names += [f"{c}__z_{a}" for c in MODEL_FEATURE_COLS for a in SHAPE_AGGREGATIONS]
    return names


def feature_groups(mode: str) -> dict[str, list[int]]:
    """Column indices grouped by source channel.

    Permutation importance over SINGLE columns is meaningless here: the six
    statistics of one channel are strongly correlated, so shuffling one leaves
    the model five substitutes and the measured drop is ~0 (observed:
    every importance below 0.0005). Shuffling a whole channel at once asks the
    question that has an answer - and the answer is one a volleyball expert can
    argue with.
    """
    names = feature_names() if mode == "absolute" else invariant_feature_names()
    groups: dict[str, list[int]] = {}
    for i, n in enumerate(names):
        base = n.split("__")[0]
        groups.setdefault(base, []).append(i)
    return groups


# ---------------------------------------------------------------------------
# 2. Dataset loading
# ---------------------------------------------------------------------------

def extract(seq_raw: np.ndarray, mode: str = "invariant") -> np.ndarray:
    """Feature extractor selected by mode: 'absolute', 'invariant', or 'both'."""
    return features_from_model_sequence(
        build_model_sequence(np.asarray(seq_raw, dtype=float),
                             target_len=SEQ_LEN),
        mode,
    )


def features_from_model_sequence(seq: np.ndarray,
                                 mode: str = "invariant") -> np.ndarray:
    """
    (29, 18) model sequence -> feature vector, in the SAME order used at fit
    time.

    This is the one function both training and live inference call, so a
    trained .joblib cannot silently receive differently-ordered features than it
    was fitted on. Guarding this is not paranoia - a train/serve feature
    mismatch has already cost this project two debugging cycles (L6, sec 12.8).
    """
    seq = np.asarray(seq, dtype=float)
    if mode == "absolute":
        return absolute_from_model_sequence(seq)
    if mode == "invariant":
        return invariant_from_model_sequence(seq)
    if mode == "both":
        return np.concatenate([absolute_from_model_sequence(seq),
                               invariant_from_model_sequence(seq)])
    raise ValueError(f"unknown feature mode {mode!r}")


def names_for(mode: str) -> list[str]:
    if mode == "absolute":
        return feature_names()
    if mode == "invariant":
        return invariant_feature_names()
    return feature_names() + invariant_feature_names()


def load_dataset(csv_dir: str, label_map: dict | None = None,
                 mode: str = "invariant"):
    """
    Read every labelled clip CSV in `csv_dir`.

    Returns (X, y, groups, files):
      X      (n, 108) float
      y      (n,)     int  index into CLASS_NAMES
      groups (n,)     str  source-video key, for leave-one-video-out
      files  (n,)     str  basename, so predictions stay traceable to a clip
    """
    import pandas as pd

    X, y, groups, files = [], [], [], []
    skipped = []

    for fname in sorted(os.listdir(csv_dir)):
        if not fname.lower().endswith(".csv"):
            continue
        path = os.path.join(csv_dir, fname)
        try:
            df = pd.read_csv(path)
        except Exception as exc:               # noqa: BLE001
            skipped.append((fname, f"unreadable: {exc}"))
            continue

        if "target_label" not in df.columns:
            skipped.append((fname, "no target_label column"))
            continue

        label = str(df["target_label"].iloc[0]).strip()
        if label_map:
            label = label_map.get(label, label)
        if label not in CLASS_NAMES:
            skipped.append((fname, f"label not in the 4 classes: {label!r}"))
            continue

        drop = [c for c in ("frame_id", "target_label") if c in df.columns]
        raw = df.drop(columns=drop).to_numpy(dtype=float)
        if raw.shape[0] < 2:
            skipped.append((fname, "fewer than 2 frames"))
            continue

        X.append(extract(raw, mode))
        y.append(CLASS_NAMES.index(label))
        groups.append(video_key(fname))
        files.append(fname)

    if not X:
        raise SystemExit(f"No usable labelled CSVs found in {csv_dir!r}")

    return (np.vstack(X), np.asarray(y, dtype=int),
            np.asarray(groups), np.asarray(files)), skipped


# ---------------------------------------------------------------------------
# 3. Model zoo
# ---------------------------------------------------------------------------

def build_models(seed: int = 0) -> dict:
    """
    Candidate classifiers, each wrapped in a Pipeline that owns its own
    imputation and (where needed) scaling.

    Putting the preprocessing INSIDE the pipeline is not a style choice: with
    cross-validation, fitting a scaler or imputer on the full dataset before
    splitting leaks test-fold statistics into training and inflates every score.
    This is the single most common way a CV number becomes indefensible.

    `class_weight='balanced'` everywhere it exists, because Delayed Support has
    16 examples against Coordinated Defense's 134 - a 8:1 imbalance that an
    unweighted learner answers by never predicting the minority class at all.
    """
    from sklearn.ensemble import (
        RandomForestClassifier, ExtraTreesClassifier,
        GradientBoostingClassifier, HistGradientBoostingClassifier,
    )
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    def scaled(clf):
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", clf),
        ])

    def bare(clf):
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", clf),
        ])

    return {
        "logistic_regression": scaled(LogisticRegression(
            max_iter=5000, class_weight="balanced", random_state=seed)),
        "linear_svm": scaled(SVC(
            kernel="linear", probability=True, class_weight="balanced",
            random_state=seed)),
        "rbf_svm": scaled(SVC(
            kernel="rbf", C=10.0, gamma="scale", probability=True,
            class_weight="balanced", random_state=seed)),
        "knn": scaled(KNeighborsClassifier(n_neighbors=5, weights="distance")),
        "mlp": scaled(MLPClassifier(
            hidden_layer_sizes=(128, 64), max_iter=3000, alpha=1e-2,
            random_state=seed)),
        "random_forest": bare(RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2, class_weight="balanced",
            n_jobs=-1, random_state=seed)),
        "extra_trees": bare(ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=2, class_weight="balanced",
            n_jobs=-1, random_state=seed)),
        "gradient_boosting": bare(GradientBoostingClassifier(
            random_state=seed)),
        "hist_gradient_boosting": bare(HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, random_state=seed)),
    }


# ---------------------------------------------------------------------------
# 4. Evaluation
# ---------------------------------------------------------------------------

def _scores(y_true, y_pred) -> dict:
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, f1_score, cohen_kappa_score,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                   zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted",
                                      zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


def cross_validate_models(X, y, groups, seed: int = 0, n_splits: int = 5):
    """
    Two protocols, because they answer two different questions and only
    reporting the friendlier one is how evaluation chapters get torn apart:

      STRATIFIED K-FOLD - clips from one video land on both sides of the split.
      Measures WITHIN-video generalisation. It is the number comparable to the
      project's existing 63.6%, and it is optimistic.

      LEAVE-ONE-VIDEO-OUT - every clip of a held-out video is unseen. Measures
      CROSS-video generalisation, which is what "will this work on next week's
      match" actually means. It is the honest number, and it will be lower.

    The GAP between them is itself a finding (sec 12.6: the model can learn which
    camera it is looking at). Report both, with the gap explained.
    """
    from sklearn.model_selection import (
        StratifiedKFold, LeaveOneGroupOut, cross_val_predict,
    )

    results = {}
    models = build_models(seed)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    logo = LeaveOneGroupOut()

    for name, model in models.items():
        row = {"model": name}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred_k = cross_val_predict(model, X, y, cv=skf, n_jobs=1)
            for k, v in _scores(y, pred_k).items():
                row[f"kfold_{k}"] = v

            n_groups = len(np.unique(groups))
            if n_groups >= 2:
                pred_l = cross_val_predict(model, X, y, cv=logo, groups=groups,
                                           n_jobs=1)
                for k, v in _scores(y, pred_l).items():
                    row[f"lovo_{k}"] = v
                row["_pred_lovo"] = pred_l
            row["_pred_kfold"] = pred_k
        results[name] = row

    return results


def calibrate(model, X, y, seed: int = 0, method: str = "sigmoid", cv: int = 5):
    """
    Wrap a fitted-able pipeline in cross-validated probability calibration.

    WHY THIS IS NOT OPTIONAL HERE
    -----------------------------
    Gradient-boosted trees fitted on 285 clips emit probabilities of 1.000 on
    almost every window - measured live: conf=1.000, entropy=0.00 bits on four
    of five sequences. That silently disables two features this project already
    depends on:

      * the anomaly / tactical-deviation flag, which fires on low confidence and
        therefore never fires at all;
      * the Shannon-entropy channel in the per-sequence CSV and the HUD, which
        the report presents as a "how unusual is this play" cue.

    An uncalibrated 1.000 is not a claim of certainty, it is an artefact of
    ensembles voting unanimously on training-set-like inputs. Calibration maps
    scores back onto frequencies that mean what the report says they mean.

    `sigmoid` (Platt) rather than `isotonic`: isotonic is non-parametric and
    needs far more data per class than the 16 examples in Delayed Support, where
    it would overfit the calibration curve itself.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import StratifiedKFold

    counts = np.bincount(y, minlength=len(CLASS_NAMES))
    smallest = int(counts[counts > 0].min())
    folds = max(2, min(cv, smallest))
    if smallest < 2:
        return model, {"calibrated": False,
                       "reason": f"smallest class has {smallest} sample(s)"}

    cal = CalibratedClassifierCV(
        model, method=method,
        cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cal.fit(X, y)
    return cal, {"calibrated": True, "method": method, "cv_folds": folds}


def confidence_profile(model, X) -> dict:
    """Summarise how confident a model is - the check that catches a model
    reporting 1.000 on everything."""
    p = model.predict_proba(X)
    conf = p.max(axis=1)
    ent = np.array([shannon_entropy(row) for row in p])
    return {
        "mean_confidence": float(conf.mean()),
        "median_confidence": float(np.median(conf)),
        "frac_confidence_above_0.99": float((conf > 0.99).mean()),
        "mean_entropy_bits": float(ent.mean()),
    }


def shannon_entropy(p) -> float:
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def majority_baseline(y) -> dict:
    """The number every reported score must beat to mean anything."""
    counts = np.bincount(y, minlength=len(CLASS_NAMES))
    pred = np.full_like(y, int(counts.argmax()))
    return _scores(y, pred)


def rule_engine_baseline(X, y) -> dict | None:
    """
    Not implemented as a re-run of `label_clips.py`: on this dataset the rule
    engine PRODUCED y, so its agreement with y is 100% by construction and
    reporting it would be meaningless. The meaningful rule-vs-model comparison
    requires human labels and lives in `evaluate_against_gold`.
    """
    return None


def evaluate_against_gold(model, csv_dir: str, gold_csv: str,
                          mode: str = "invariant") -> dict:
    """
    TIER 2 - the number that answers the examiner.

    `gold_csv` is produced by `fyp/annotate.py`: columns `clip,label` where
    `label` is a human judgement. Both the rule engine's label (already in the
    clip CSV) and the model's prediction are scored against it.

    If the model beats the rules against human judgement, it has denoised its
    teacher rather than merely copied it - and that is a defensible,
    literature-backed claim (weak supervision). If it does not, say so: the
    honest negative result is worth more than an unfalsifiable positive one.
    """
    import pandas as pd

    gold = pd.read_csv(gold_csv)
    if not {"clip", "label"}.issubset(gold.columns):
        raise SystemExit("gold CSV needs columns: clip,label")

    y_h, y_rule, feats, used = [], [], [], []
    for _, r in gold.iterrows():
        human = str(r["label"]).strip()
        if human not in CLASS_NAMES:
            continue
        path = os.path.join(csv_dir, str(r["clip"]).strip())
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if "target_label" not in df.columns:
            continue
        rule = str(df["target_label"].iloc[0]).strip()
        if rule not in CLASS_NAMES:
            continue
        drop = [c for c in ("frame_id", "target_label") if c in df.columns]
        # MUST use the same feature mode the model was fitted with; using the
        # default here silently feeds a 132-column model a 108-column vector.
        feats.append(extract(df.drop(columns=drop).to_numpy(dtype=float), mode))
        y_h.append(CLASS_NAMES.index(human))
        y_rule.append(CLASS_NAMES.index(rule))
        used.append(str(r["clip"]).strip())

    if not feats:
        raise SystemExit("No gold rows matched a clip CSV with a rule label.")

    y_h = np.asarray(y_h)
    y_rule = np.asarray(y_rule)
    y_model = model.predict(np.vstack(feats))

    return {
        "n_gold": int(len(y_h)),
        "rules_vs_human": _scores(y_h, y_rule),
        "model_vs_human": _scores(y_h, y_model),
        "model_vs_rules": _scores(y_rule, y_model),
        "clips": used,
    }


def grouped_permutation_importance(model_name, X, y, groups, mode: str,
                                   seed: int = 0, n_repeats: int = 5,
                                   top: int = 20):
    """
    Channel-level importance, measured OUT OF SAMPLE.

    Two corrections over the textbook call, both of which changed the answer on
    this dataset:

    1. SHUFFLE THE CHANNEL, NOT THE COLUMN. The six statistics of a channel are
       strongly correlated, so removing one leaves five substitutes and the
       measured drop collapses to ~0 - observed here, where every single-column
       importance came out below 0.0005 and the ranking was pure noise. The
       block permutation removes the information rather than one encoding of it.

    2. MEASURE ON HELD-OUT DATA. Importance computed on the data the model was
       fitted to is close to meaningless for a high-capacity learner: fitted on
       all 285 clips the winning model scores in-sample macro-F1 1.000, i.e. it
       has memorised, and it can answer from any surviving column no matter what
       is shuffled. Every channel then looks unimportant. Here the model is
       refitted per leave-one-video-out fold and permuted on the held-out video,
       so the number means "how much CROSS-VIDEO performance depended on this
       measurement" - which is the only version worth showing an expert.
    """
    from sklearn.metrics import f1_score
    from sklearn.model_selection import LeaveOneGroupOut

    rng = np.random.default_rng(seed)
    chan_cols = feature_groups(mode)
    logo = LeaveOneGroupOut()

    base_scores: list[float] = []
    drops: dict[str, list[float]] = {c: [] for c in chan_cols}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tr, te in logo.split(X, y, groups):
            mdl = build_models(seed)[model_name]
            mdl.fit(X[tr], y[tr])
            base = f1_score(y[te], mdl.predict(X[te]),
                            average="macro", zero_division=0)
            base_scores.append(base)

            for channel, cols in chan_cols.items():
                for _ in range(n_repeats):
                    Xp = X[te].copy()
                    Xp[:, cols] = Xp[rng.permutation(len(Xp))][:, cols]
                    drops[channel].append(
                        base - f1_score(y[te], mdl.predict(Xp),
                                        average="macro", zero_division=0))

    out = [{
        "channel": c,
        "n_columns": len(chan_cols[c]),
        "importance_mean": float(np.mean(v)),
        "importance_std": float(np.std(v)),
    } for c, v in drops.items()]
    out.sort(key=lambda d: -d["importance_mean"])

    return {
        "baseline_macro_f1": float(np.mean(base_scores)),
        "protocol": "leave-one-video-out, permuted on the held-out video",
        "channels": out[:top],
    }


# ---------------------------------------------------------------------------
# 5. Reporting
# ---------------------------------------------------------------------------

def confusion_table(y_true, y_pred) -> str:
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))
    short = [c[:14] for c in CLASS_NAMES]
    w = max(len(s) for s in short) + 2

    lines = [" " * w + "".join(f"{s[:9]:>11}" for s in short) + "   <- predicted"]
    for i, row in enumerate(cm):
        total = row.sum()
        recall = row[i] / total if total else 0.0
        lines.append(f"{short[i]:<{w}}"
                     + "".join(f"{v:>11d}" for v in row)
                     + f"   recall {row[i]}/{total} = {recall:.2f}")
    return "\n".join(lines)


def per_class_report(y_true, y_pred) -> str:
    from sklearn.metrics import classification_report
    return classification_report(
        y_true, y_pred, labels=range(len(CLASS_NAMES)),
        target_names=CLASS_NAMES, zero_division=0, digits=3,
    )



def _run_ablation(args) -> int:
    """Same models, same folds, three feature blocks - isolates the effect of
    the representation from the effect of the learner."""
    print("\n" + "=" * 74)
    print("FEATURE-BLOCK ABLATION")
    print("=" * 74)
    print("\nAbsolute features are court-cm aggregates. Without a per-camera")
    print("homography those centimetres are not comparable between videos, so a")
    print("learner can use them to identify the CAMERA rather than the tactic.")
    print("Invariant features are dimensionless ratios plus per-clip z-scored")
    print("temporal shape, both of which survive an unknown scale factor.")
    print("\nIf LOVO improves under 'invariant', the per-video signature reported")
    print("in sec 12.6 was substantially a units problem, not a data-size problem.\n")

    rows = []
    for mode in ("absolute", "invariant", "both"):
        (X, y, groups, _f), _s = load_dataset(args.csv_dir, mode=mode)
        res = cross_validate_models(X, y, groups, seed=args.seed,
                                    n_splits=args.folds)
        best = max(res, key=lambda k: res[k].get("lovo_macro_f1", -1))
        rows.append((mode, X.shape[1], best,
                     res[best].get("kfold_macro_f1", float("nan")),
                     res[best].get("lovo_macro_f1", float("nan")),
                     res[best].get("lovo_accuracy", float("nan")),
                     res[best].get("lovo_cohen_kappa", float("nan"))))
        print(f"  {mode:<10} n_feat={X.shape[1]:<4} best={best:<24} "
              f"kfold_F1={rows[-1][3]:.3f}  LOVO_F1={rows[-1][4]:.3f}  "
              f"kappa={rows[-1][6]:+.3f}")

    print("\n" + "-" * 74)
    hdr = f"{'features':<12}{'n':>5}{'best model':>26}{'kfoldF1':>9}{'lovoF1':>9}{'kappa':>8}"
    print(hdr)
    print("-" * len(hdr))
    for mode, n, best, kf, lf, la, kp in rows:
        print(f"{mode:<12}{n:>5}{best:>26}{kf:>9.3f}{lf:>9.3f}{kp:>8.3f}")

    a = dict((r[0], r[4]) for r in rows)
    delta = a["invariant"] - a["absolute"]
    print(f"\ninvariant - absolute, LOVO macro-F1: {delta:+.3f}")
    if delta > 0.02:
        print("  -> Scale-invariant features generalise better across cameras.")
        print("     Report this ablation: it converts 'our model does not transfer'")
        print("     into 'our model transfers once the features are dimensionless'.")
    else:
        print("  -> No material gain. The cross-video gap is NOT mainly a units")
        print("     problem on this data; report that honestly and look to label")
        print("     quality and sample size instead.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Learned tactical classifier (scikit-learn).")
    ap.add_argument("csv_dir", help="Directory of labelled clip CSVs.")
    ap.add_argument("--output-dir", default="ml_results")
    ap.add_argument("--gold", default=None,
                    help="Human-labelled CSV (clip,label) for the Tier-2 test.")
    ap.add_argument("--save-model", default=None,
                    help="Path to persist the winning pipeline (.joblib).")
    ap.add_argument("--features", choices=("absolute", "invariant", "both"),
                    default="invariant",
                    help="Feature block. 'absolute' = raw court-cm aggregates "
                         "(camera-dependent under the linear fallback); "
                         "'invariant' = dimensionless ratios + per-clip "
                         "z-scored temporal shape (default); 'both' = union.")
    ap.add_argument("--ablate-features", action="store_true",
                    help="Run all three feature blocks and print the "
                         "comparison - the ablation an examiner will ask for.")
    ap.add_argument("--no-calibrate", action="store_true",
                    help="Skip probability calibration. Not recommended: the "
                         "tree ensembles emit 1.000 confidence on nearly every "
                         "window, which disables the entropy/anomaly channel.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--select-by", default="lovo_macro_f1",
                    help="Metric used to pick the winner "
                         "(default lovo_macro_f1 - the honest one).")
    args = ap.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 74)
    print("LEARNED TACTICAL CLASSIFIER")
    print("=" * 74)

    if args.ablate_features:
        return _run_ablation(args)

    (X, y, groups, files), skipped = load_dataset(args.csv_dir,
                                                  mode=args.features)
    print(f"\nLoaded {len(X)} clips, {X.shape[1]} features "
          f"(mode={args.features})")
    if skipped:
        print(f"  skipped {len(skipped)} files "
              f"(first: {skipped[0][0]} - {skipped[0][1]})")

    print("\nClass distribution:")
    for i, c in enumerate(CLASS_NAMES):
        n = int((y == i).sum())
        print(f"  {c:<22}: {n:>4}  ({100*n/len(y):>4.1f}%)")
    print("\nSource videos (leave-one-video-out folds):")
    for g in np.unique(groups):
        print(f"  {g:<10}: {int((groups == g).sum()):>4} clips")

    base = majority_baseline(y)
    print(f"\nMajority-class baseline: accuracy {base['accuracy']:.3f}  "
          f"macro-F1 {base['macro_f1']:.3f}")
    print("  Every number below must beat this to carry any information.")

    print("\n" + "=" * 74)
    print("CROSS-VALIDATION  (k-fold = optimistic, LOVO = honest)")
    print("=" * 74)
    results = cross_validate_models(X, y, groups, seed=args.seed,
                                    n_splits=args.folds)

    hdr = (f"{'model':<24}{'kf_acc':>8}{'kf_F1':>8}"
           f"{'lovo_acc':>10}{'lovo_F1':>9}{'lovo_bal':>10}{'kappa':>8}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for name, r in sorted(results.items(),
                          key=lambda kv: -kv[1].get(args.select_by, -1)):
        print(f"{name:<24}"
              f"{r.get('kfold_accuracy', float('nan')):>8.3f}"
              f"{r.get('kfold_macro_f1', float('nan')):>8.3f}"
              f"{r.get('lovo_accuracy', float('nan')):>10.3f}"
              f"{r.get('lovo_macro_f1', float('nan')):>9.3f}"
              f"{r.get('lovo_balanced_accuracy', float('nan')):>10.3f}"
              f"{r.get('lovo_cohen_kappa', float('nan')):>8.3f}")

    best_name = max(results, key=lambda k: results[k].get(args.select_by, -1))
    best = results[best_name]
    print(f"\nWinner by {args.select_by}: {best_name}")

    pred_key = "_pred_lovo" if "_pred_lovo" in best else "_pred_kfold"
    y_pred = best[pred_key]
    proto = "leave-one-video-out" if pred_key == "_pred_lovo" else "k-fold"

    print(f"\nConfusion matrix ({proto}, {best_name}):")
    print(confusion_table(y, y_pred))
    print(f"\nPer-class report ({proto}):")
    print(per_class_report(y, y_pred))

    print("Fitting the winner on all data for importance + deployment...")
    model = build_models(args.seed)[best_name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)

    raw_profile = confidence_profile(model, X)
    cal_info = {"calibrated": False, "reason": "--no-calibrate"}
    if not args.no_calibrate:
        model, cal_info = calibrate(model, X, y, seed=args.seed)
    cal_profile = confidence_profile(model, X)

    print("\nProbability calibration:")
    print(f"  {cal_info}")
    print(f"  {'':<22}{'uncalibrated':>14}{'calibrated':>13}")
    for k in ("mean_confidence", "frac_confidence_above_0.99",
              "mean_entropy_bits"):
        print(f"  {k:<22}{raw_profile[k]:>14.3f}{cal_profile[k]:>13.3f}")
    if raw_profile["frac_confidence_above_0.99"] > 0.5 >= cal_profile["frac_confidence_above_0.99"]:
        print("  -> Calibration restored a usable confidence range, so the")
        print("     entropy channel and the anomaly flag work again.")

    print("\nChannel importance (macro-F1 lost when the whole channel is "
          "shuffled):")
    imp = grouped_permutation_importance(best_name, X, y, groups,
                                          args.features, seed=args.seed)
    print(f"  protocol: {imp['protocol']}")
    print(f"  held-out baseline macro-F1: {imp['baseline_macro_f1']:.3f}")
    for d in imp["channels"][:15]:
        bar = "#" * max(int(d["importance_mean"] * 120), 0)
        print(f"  {d['channel']:<22} {d['importance_mean']:+.4f} "
              f"+/- {d['importance_std']:.4f}  {bar}")

    gold_report = None
    if args.gold:
        print("\n" + "=" * 74)
        print("TIER 2 - scored against HUMAN labels")
        print("=" * 74)
        gold_report = evaluate_against_gold(model, args.csv_dir, args.gold,
                                            mode=args.features)
        r_h = gold_report["rules_vs_human"]["macro_f1"]
        m_h = gold_report["model_vs_human"]["macro_f1"]
        print(f"\n  gold clips                : {gold_report['n_gold']}")
        print(f"  RULE ENGINE vs human macro-F1 : {r_h:.3f}")
        print(f"  MODEL       vs human macro-F1 : {m_h:.3f}")
        print(f"  model vs rules  macro-F1      : "
              f"{gold_report['model_vs_rules']['macro_f1']:.3f}")
        if m_h > r_h:
            print(f"\n  -> The model outscores its own teacher by "
                  f"{m_h - r_h:+.3f} macro-F1 against human judgement.\n"
                  f"     It has denoised the rules rather than copied them - the\n"
                  f"     standard weak-supervision result, and the answer to\n"
                  f"     'isn't your ML model just your rules?'")
        else:
            print(f"\n  -> The model does NOT beat the rules against humans "
                  f"({m_h - r_h:+.3f}).\n"
                  f"     Report this honestly: on this data the learned model is\n"
                  f"     a faster, differentiable restatement of the heuristic,\n"
                  f"     and Tier 3 (training ON human labels) is required.")
    else:
        print("\n" + "!" * 74)
        print("NO GOLD SET SUPPLIED - every number above measures AGREEMENT WITH")
        print("THE RULE ENGINE, not tactical correctness. Do not present these as")
        print("accuracy to a volleyball expert. Build the gold set:")
        print("    python fyp/annotate.py dataset/dataset --out gold_labels.csv")
        print("then re-run with  --gold gold_labels.csv")
        print("!" * 74)

    summary = {
        "n_clips": int(len(X)),
        "n_features": int(X.shape[1]),
        "class_counts": {c: int((y == i).sum()) for i, c in enumerate(CLASS_NAMES)},
        "majority_baseline": base,
        "winner": best_name,
        "selected_by": args.select_by,
        "models": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                   for k, v in results.items()},
        "channel_importance": imp,
        "feature_mode": args.features,
        "gold": gold_report,
        "calibration": cal_info,
        "confidence_uncalibrated": raw_profile,
        "confidence_calibrated": cal_profile,
        "label_provenance": (
            "Labels come from label_clips.py (rule engine). Cross-validation "
            "scores measure agreement with that teacher, not tactical accuracy. "
            "Only the 'gold' block compares against human judgement."
        ),
    }
    out_json = os.path.join(args.output_dir, "ml_results.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nSummary written to {out_json}")

    if args.save_model:
        try:
            import joblib
        except ImportError:
            print("[WARNING] joblib not installed; model not saved.",
                  file=sys.stderr)
        else:
            joblib.dump(
                {"pipeline": model,
                 "class_names": CLASS_NAMES,
                 "feature_names": names_for(args.features),
                 "aggregations": list(AGGREGATIONS),
                 "feature_mode": args.features,
                 "feature_version": __import__("features").FEATURE_VERSION,
                 "model_name": best_name},
                args.save_model,
            )
            print(f"Model saved to {args.save_model}")
            print("  Serve it live with:")
            print(f"    python fyp/pipeline.py video.mp4 {args.save_model} …")

    return 0


if __name__ == "__main__":   # pragma: no cover
    raise SystemExit(main())
