"""Unit tests for analytics.py - NumPy/pandas only, no torch/YOLO."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics import (  # noqa: E402
    clip_metrics, METRIC_COLS,
    fit_deviation_model, DEVIATION_FEATURES,
    confidence_label,
)


def _synth_clip(spread=300.0, motion=0.0, seed=0):
    """Make a raw (29,14) clip: 6 players, optional coherent motion."""
    rng = np.random.default_rng(seed)
    T = 29
    base = rng.uniform(200, 1600, size=(6, 2))
    raw = np.zeros((T, 14), dtype=np.float32)
    for t in range(T):
        pos = base + motion * t  # all drift together if motion>0
        raw[t, 2:] = pos.reshape(-1)
        raw[t, 0:2] = [900.0, 450.0]  # ball present, on court
    return raw


def test_clip_metrics_keys_and_finite():
    m = clip_metrics(_synth_clip())
    for c in METRIC_COLS:
        assert c in m
    assert m["n_frames"] == 29
    assert 0.0 <= m["ball_present_frac"] <= 1.0
    assert np.isfinite(m["mean_spacing_cm"])


def test_speed_scale_is_cms():
    # players moving 5 cm/frame -> 150 cm/s.
    raw = _synth_clip(motion=np.array([5.0, 0.0]))
    m = clip_metrics(raw)
    assert 120 < m["top2_speed_cms"] < 180, m["top2_speed_cms"]


def test_confidence_abstention():
    names = ["A", "B", "C", "D"]
    lab, conf = confidence_label([0.1, 0.2, 0.3, 0.4], names, tau=0.5)
    assert lab == "Uncertain" and abs(conf - 0.4) < 1e-6
    lab2, conf2 = confidence_label([0.05, 0.05, 0.1, 0.8], names, tau=0.5)
    assert lab2 == "D" and abs(conf2 - 0.8) < 1e-6


def test_deviation_higher_for_outliers():
    # Reference cluster = tight; outliers = far -> larger Mahalanobis distance.
    rng = np.random.default_rng(1)
    ref = pd.DataFrame(rng.normal(0, 1, size=(60, len(DEVIATION_FEATURES))),
                       columns=DEVIATION_FEATURES)
    out = pd.DataFrame(rng.normal(8, 1, size=(10, len(DEVIATION_FEATURES))),
                       columns=DEVIATION_FEATURES)
    allrows = pd.concat([ref, out], ignore_index=True)
    mask = np.array([True] * 60 + [False] * 10)
    model = fit_deviation_model(allrows, mask)
    ref_scores = np.mean([model.score(r) for r in ref.to_dict("records")])
    out_scores = np.mean([model.score(r) for r in out.to_dict("records")])
    assert out_scores > ref_scores * 3, (ref_scores, out_scores)


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print("  PASS  " + fn.__name__)
    print("\n{}/{} tests passed.".format(len(fns), len(fns)))


if __name__ == "__main__":
    _run_all()
