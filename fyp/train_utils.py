"""
train_utils.py - torch-free training helpers (unit-testable without a GPU).

class_weights_from_counts guards against the collapse observed on the first
full Kaggle run: the Unclassified class had ZERO training samples, but the old
inverse-frequency formula `K/(counts+1)` still gave it a ~100x weight relative
to the majority class. torch's CrossEntropyLoss applies class weights to the
label-smoothing target mass too (eps/K on EVERY class), so the loss rewarded
probability on the empty class more than on the true class -> the model
predicted Unclassified for everything (train_acc = 0.000, val collapse).

Rules implemented here:
  * absent classes (count == 0) get weight EXACTLY 0 - no gradient pull, and
    their label-smoothing mass contributes nothing;
  * present classes get inverse-frequency weights, mean-normalised to 1 so the
    effective learning rate is preserved;
  * the minority boost is capped (default 4x) so a tiny class cannot dominate.
"""
from __future__ import annotations

import numpy as np


def class_weights_from_counts(counts, cap: float = 4.0) -> np.ndarray:
    """Inverse-frequency class weights; empty classes get exactly 0.

    Parameters
    ----------
    counts : per-class training-sample counts, shape (K,)
    cap    : maximum weight after mean-normalisation over present classes.
    """
    counts = np.asarray(counts, dtype=float).reshape(-1)
    weights = np.zeros_like(counts)
    present = counts > 0
    if not present.any():
        return weights
    weights[present] = 1.0 / counts[present]
    weights[present] = weights[present] / weights[present].mean()
    return np.minimum(weights, float(cap))


__all__ = ["class_weights_from_counts"]


import re as _re


def video_key(fname: str) -> str:
    """Source-video key from a clip/CSV filename.

    'Coordinated_Attack__videoplayback (1)_clip_002.csv' -> '(1)'
    'Delayed_Support__videoplayback_clip_306.csv'        -> '(plain)'
    """
    m = _re.search(r"videoplayback(?:\s*(\(\d+\)))?_clip", str(fname))
    if not m:
        return "?"
    return m.group(1) if m.group(1) else "(plain)"


def holdout_indices(files, test_key: str):
    """Split indices into (test, rest) by source-video key.

    This is the leave-one-video-out protocol: the ENTIRE held-out video is
    unseen at training time, so test metrics measure cross-video
    generalisation instead of within-video memorisation."""
    test_key = test_key if test_key.startswith("(") else f"({test_key})"
    test, rest = [], []
    for i, f in enumerate(files):
        (test if video_key(f) == test_key else rest).append(i)
    return test, rest
