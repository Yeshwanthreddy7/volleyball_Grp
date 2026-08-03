"""
Tests for the learned tactical classifier (fyp/ml_classifier.py) and the blind
annotator (fyp/annotate.py).

The properties locked here are the ones whose violation produced a wrong NUMBER
rather than a crash - the failure mode this project keeps hitting.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ml_classifier as mc  # noqa: E402
import annotate as an  # noqa: E402
from features import MODEL_FEATURE_COLS  # noqa: E402


def _seq(n=29, d=14, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(500.0, 50.0, size=(n, d))


# --------------------------------------------------------------------------
# feature contract
# --------------------------------------------------------------------------

def test_absolute_block_has_one_column_per_channel_and_statistic():
    x = mc.sequence_features(_seq())
    assert x.shape == (len(MODEL_FEATURE_COLS) * len(mc.AGGREGATIONS),)
    assert len(mc.feature_names()) == len(x)


def test_invariant_block_names_match_width():
    x = mc.invariant_sequence_features(_seq())
    assert len(mc.invariant_feature_names()) == len(x)


def test_both_mode_is_the_concatenation_in_order():
    s = _seq()
    a = mc.sequence_features(s)
    i = mc.invariant_sequence_features(s)
    b = mc.extract(s, "both")
    assert np.allclose(b, np.concatenate([a, i]), equal_nan=True)
    assert len(mc.names_for("both")) == len(b)


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        mc.extract(_seq(), "nonsense")


def test_train_and_serve_share_one_extractor():
    """The train/serve mismatch guard. `pipeline.py` hands the classifier a
    (29,18) MODEL sequence while training starts from (29,14) raw rows; both
    must funnel into the same aggregation in the same column order, or a saved
    model silently scores garbage (the L6 / sec 12.8 failure class)."""
    from features import build_model_sequence

    raw = _seq()
    model_seq = build_model_sequence(raw, target_len=mc.SEQ_LEN)
    for mode in ("absolute", "invariant", "both"):
        assert np.allclose(mc.extract(raw, mode),
                           mc.features_from_model_sequence(model_seq, mode),
                           equal_nan=True)


# --------------------------------------------------------------------------
# scale invariance - the property the whole invariant block exists for
# --------------------------------------------------------------------------

def test_invariant_features_survive_a_camera_scale_change():
    """Under the linear pixel->court fallback, two cameras give the same play
    different centimetres. Ratio and z-scored-shape features must not move."""
    raw = _seq()
    scaled = raw * 1.7                      # same play, different court scale

    a = mc.invariant_sequence_features(raw)
    b = mc.invariant_sequence_features(scaled)
    same = np.isclose(a, b, rtol=1e-3, atol=1e-3) | (np.isnan(a) & np.isnan(b))
    assert same.mean() > 0.8, (
        f"only {100*same.mean():.0f}% of invariant features survived a scale "
        "change; the block is not doing its job")


def test_absolute_features_do_move_under_a_scale_change():
    """Control for the test above: if absolutes were also scale-stable, the
    ablation would be measuring nothing."""
    raw = _seq()
    a = mc.sequence_features(raw)
    b = mc.sequence_features(raw * 1.7)
    moved = ~np.isclose(a, b, rtol=1e-3, atol=1e-3)
    assert moved.mean() > 0.5


# --------------------------------------------------------------------------
# numerical robustness
# --------------------------------------------------------------------------

def test_slope_is_signed_and_scaled_per_frame():
    assert mc._slope(np.arange(29, dtype=float)) == pytest.approx(1.0)
    assert mc._slope(-np.arange(29, dtype=float)) == pytest.approx(-1.0)
    assert mc._slope(np.full(29, 3.0)) == pytest.approx(0.0)


def test_slope_handles_nan_and_degenerate_input():
    x = np.full(29, np.nan)
    assert mc._slope(x) == 0.0
    x[3] = 1.0
    assert mc._slope(x) == 0.0            # a single point defines no slope


def test_features_are_finite_even_when_the_ball_is_never_seen():
    """42% of frames had no ball historically; an all-zero ball channel must not
    produce inf from a ratio denominator."""
    raw = _seq()
    raw[:, 0:2] = 0.0                      # ball_x, ball_y absent
    for mode in ("absolute", "invariant", "both"):
        v = mc.extract(raw, mode)
        assert not np.isinf(v).any(), f"{mode} produced inf"


def test_ratio_features_are_clipped_against_a_zero_denominator():
    raw = np.zeros((29, 14))               # everything collapsed to one point
    v = mc.invariant_sequence_features(raw)
    assert np.all(np.abs(v[np.isfinite(v)]) <= 50.0 + 1e-6)


# --------------------------------------------------------------------------
# grouping / importance
# --------------------------------------------------------------------------

def test_feature_groups_partition_every_column_exactly_once():
    for mode in ("absolute", "invariant"):
        groups = mc.feature_groups(mode)
        cols = sorted(c for v in groups.values() for c in v)
        assert cols == list(range(len(mc.names_for(mode))))


def test_absolute_groups_are_the_18_channels():
    assert set(mc.feature_groups("absolute")) == set(MODEL_FEATURE_COLS)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def test_majority_baseline_matches_the_dominant_class_share():
    y = np.array([1] * 7 + [0] * 3)
    assert mc.majority_baseline(y)["accuracy"] == pytest.approx(0.7)


def test_shannon_entropy_endpoints():
    assert mc.shannon_entropy([1.0, 0, 0, 0]) == pytest.approx(0.0)
    assert mc.shannon_entropy([0.25] * 4) == pytest.approx(2.0)


def test_confusion_table_reports_per_class_recall():
    y = np.array([0, 0, 1, 1])
    table = mc.confusion_table(y, np.array([0, 1, 1, 1]))
    assert "recall 1/2" in table and "recall 2/2" in table


# --------------------------------------------------------------------------
# annotator - the blindness properties that make the gold set admissible
# --------------------------------------------------------------------------

def test_clip_path_maps_to_the_rule_label_csv_name():
    p = os.path.join("dataset", "Coordinated_Attack", "videoplayback (1)_clip_002.mp4")
    assert an.csv_name_for(p) == "Coordinated_Attack__videoplayback (1)_clip_002.csv"


def test_sampling_is_balanced_across_rule_classes(tmp_path):
    """Rule labels run 134/90/45/16. Proportional sampling would put ~4 Delayed
    Support clips in the gold set - too few to estimate a per-class score."""
    for folder in an.FOLDER_TO_LABEL:
        d = tmp_path / folder
        d.mkdir()
        for i in range(10):
            (d / f"clip_{i}.mp4").write_bytes(b"")

    got = an.collect_clips(str(tmp_path), per_class=4, seed=0)
    per = {}
    for path, _ in got:
        per[os.path.basename(os.path.dirname(path))] = \
            per.get(os.path.basename(os.path.dirname(path)), 0) + 1
    assert set(per.values()) == {4}


def test_sampling_shuffles_across_classes_so_order_leaks_nothing(tmp_path):
    """Even with the folder name hidden, class-ordered playback would tell the
    annotator the rule label."""
    for folder in an.FOLDER_TO_LABEL:
        d = tmp_path / folder
        d.mkdir()
        for i in range(10):
            (d / f"clip_{i}.mp4").write_bytes(b"")

    order = [os.path.basename(os.path.dirname(p))
             for p, _ in an.collect_clips(str(tmp_path), per_class=10, seed=0)]
    runs = sum(1 for a, b in zip(order, order[1:]) if a != b)
    assert runs > len(order) // 3, "clips are still grouped by rule class"


def test_unclear_is_a_recordable_outcome_not_a_forced_class():
    assert an.UNCLEAR not in an.CLASS_NAMES


def test_annotator_classes_match_the_classifier_classes():
    assert an.CLASS_NAMES == mc.CLASS_NAMES


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-q"]))
