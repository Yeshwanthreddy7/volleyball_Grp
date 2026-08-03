"""
Unit tests for the bulletproofing upgrades:

  * occlusion gap interpolation (spec §2C)
  * kinematic features: top-2/bottom-4 speed differential + sync (spec §1B)
  * the v2 model-input contract (18 features, versioned)
  * court geometry: corner ordering, homography, mask filtering (spec §3)
  * Shannon entropy + anomaly threshold (spec §4C)
  * class-0 activity gate (spec §4B)

Run:  python -m pytest fyp/tests/test_upgrades.py -q
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from features import (  # noqa: E402
    FEATURE_VERSION,
    KINEMATIC_COLS,
    MODEL_INPUT_DIM,
    build_model_sequence,
    interpolate_gaps,
    kinematic_features,
)


# ---------------------------------------------------------------------------
# interpolate_gaps
# ---------------------------------------------------------------------------

def _line_track(T: int) -> np.ndarray:
    """One player moving on a straight line: x = 10t, y = 5t."""
    tr = np.zeros((T, 1, 2))
    tr[:, 0, 0] = 10.0 * np.arange(T)
    tr[:, 0, 1] = 5.0 * np.arange(T)
    return tr


def test_short_gap_is_interpolated_linearly():
    tr = _line_track(20)
    tr[5:10, 0, :] = np.nan               # 5-frame occlusion
    out = interpolate_gaps(tr, max_gap=15)
    assert np.allclose(out[:, 0, 0], 10.0 * np.arange(20))
    assert np.allclose(out[:, 0, 1], 5.0 * np.arange(20))


def test_long_gap_is_not_bridged():
    tr = _line_track(40)
    tr[5:25, 0, :] = np.nan               # 20-frame gap > max_gap=15
    out = interpolate_gaps(tr, max_gap=15)
    assert np.isnan(out[10, 0, 0])        # still missing


def test_edge_gaps_are_not_extrapolated():
    tr = _line_track(10)
    tr[:3, 0, :] = np.nan                 # leading gap: no left anchor
    tr[8:, 0, :] = np.nan                 # trailing gap: no right anchor
    out = interpolate_gaps(tr, max_gap=15)
    assert np.isnan(out[0, 0, 0]) and np.isnan(out[9, 0, 0])


def test_exactly_15_frame_gap_is_bridged():
    tr = _line_track(20)
    tr[2:17, 0, :] = np.nan               # exactly 15 missing frames
    out = interpolate_gaps(tr, max_gap=15)
    assert np.isfinite(out[:, 0, 0]).all()


# ---------------------------------------------------------------------------
# kinematic features (speed differentials + instantaneous sync)
# ---------------------------------------------------------------------------

def test_speed_top2_bot4_separates_attackers_from_base():
    # 2 fast players (20 cm/frame) + 4 slow players (2 cm/frame)
    T, K = 10, 6
    tr = np.zeros((T, K, 2))
    for t in range(T):
        tr[t, 0] = [20.0 * t, 0]
        tr[t, 1] = [0, 20.0 * t]
        for k in range(2, 6):
            tr[t, k] = [100 + k * 50 + 2.0 * t, 300]
    kin = kinematic_features(tr)
    assert kin.shape == (T, len(KINEMATIC_COLS))
    assert np.allclose(kin[1:, 0], 20.0)          # top-2 mean
    assert np.allclose(kin[1:, 1], 2.0)           # bottom-4 mean
    assert np.allclose(kin[1:, 2], 18.0)          # differential


def test_sync_inst_is_one_for_parallel_motion():
    T, K = 6, 4
    tr = np.zeros((T, K, 2))
    for k in range(K):
        tr[:, k, 0] = 100 * k + 7.0 * np.arange(T)   # same velocity vector
        tr[:, k, 1] = 50.0
    kin = kinematic_features(tr)
    assert np.allclose(kin[1:, 3], 1.0, atol=1e-6)


def test_sync_inst_is_negative_for_opposing_motion():
    T = 6
    tr = np.zeros((T, 2, 2))
    tr[:, 0, 0] = 10.0 * np.arange(T)
    tr[:, 1, 0] = 1000 - 10.0 * np.arange(T)
    kin = kinematic_features(tr)
    assert np.allclose(kin[1:, 3], -1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# v2 model-input contract
# ---------------------------------------------------------------------------

def test_build_model_sequence_shape_and_version():
    assert FEATURE_VERSION == "perm_invariant_v2"
    raw = np.random.uniform(0, 900, size=(29, 14)).astype(np.float32)
    seq = build_model_sequence(raw, target_len=29)
    assert seq.shape == (29, MODEL_INPUT_DIM)
    assert MODEL_INPUT_DIM == 18
    assert np.isfinite(seq).all()          # model input must be NaN-free


def test_build_model_sequence_pads_short_clips():
    raw = np.random.uniform(0, 900, size=(17, 14)).astype(np.float32)
    seq = build_model_sequence(raw, target_len=29)
    assert seq.shape == (29, MODEL_INPUT_DIM)


# ---------------------------------------------------------------------------
# court geometry (spec §3)
# ---------------------------------------------------------------------------

def test_order_corners():
    pytest.importorskip("cv2")
    from court import order_corners
    pts = np.array([[100, 400], [900, 50], [50, 60], [880, 420]], float)
    ordered = order_corners(pts)
    tl, tr, br, bl = ordered
    assert tl[0] < tr[0] and bl[0] < br[0]     # left of right
    assert tl[1] < bl[1] and tr[1] < br[1]     # above bottom


def test_manual_homography_maps_corners_to_court():
    pytest.importorskip("cv2")
    from court import COURT_H_CM, COURT_W_CM, CourtCalibrator
    corners = [(100, 100), (1180, 100), (1180, 620), (100, 620)]
    cal = CourtCalibrator(1280, 720, manual_corners=corners)
    assert cal.calibrated
    x, y = cal.pixel_to_court(100, 100)
    assert abs(x) < 1 and abs(y) < 1
    x, y = cal.pixel_to_court(1180, 620)
    assert abs(x - COURT_W_CM) < 1 and abs(y - COURT_H_CM) < 1


def test_court_mask_drops_outside_detections():
    pytest.importorskip("cv2")
    from court import CourtCalibrator
    corners = [(100, 100), (1180, 100), (1180, 620), (100, 620)]
    cal = CourtCalibrator(1280, 720, manual_corners=corners)
    xyxy = np.array([
        [500, 300, 560, 500],     # foot (530, 500) inside
        [10, 10, 60, 90],         # foot (35, 90) outside – referee/crowd
    ], dtype=float)
    conf = np.array([0.9, 0.8])
    kept_xyxy, kept_conf = cal.filter_detections(xyxy, conf)
    assert len(kept_xyxy) == 1
    assert kept_xyxy[0, 0] == 500


def test_uncalibrated_mask_is_noop():
    pytest.importorskip("cv2")
    from court import CourtCalibrator
    cal = CourtCalibrator(1280, 720)
    xyxy = np.array([[10, 10, 60, 90]], dtype=float)
    conf = np.array([0.8])
    kept, _ = cal.filter_detections(xyxy, conf)
    assert len(kept) == 1


# ---------------------------------------------------------------------------
# entropy + anomaly (spec §4C)
# ---------------------------------------------------------------------------

def test_shannon_entropy():
    from interfaces import shannon_entropy_bits
    assert abs(shannon_entropy_bits(np.array([1, 0, 0, 0, 0.0]))) < 1e-9
    uniform = np.full(5, 0.2)
    assert abs(shannon_entropy_bits(uniform) - np.log2(5)) < 1e-9


# ---------------------------------------------------------------------------
# class-0 activity gate (spec §4B)  – needs torch (pipeline import chain)
# ---------------------------------------------------------------------------

def test_activity_gate_routes_dead_ball_to_class0():
    pytest.importorskip("torch")
    pytest.importorskip("cv2")
    from pipeline import _activity_gate

    # Nearly static players -> dead ball
    raw = np.zeros((29, 14), dtype=np.float32)
    for k in range(6):
        raw[:, 2 + 2 * k] = 300 + 100 * k          # constant x
        raw[:, 3 + 2 * k] = 600                     # constant y
    is_dead, gate = _activity_gate(raw)
    assert is_dead
    assert gate["gate_mean_speed_cm_frame"] < 2.0

    # Fast, populated play -> not dead
    raw2 = raw.copy()
    for t in range(29):
        for k in range(6):
            raw2[t, 2 + 2 * k] = 300 + 100 * k + 8.0 * t
    is_dead2, _ = _activity_gate(raw2)
    assert not is_dead2


def test_gate_flags_too_few_players():
    pytest.importorskip("torch")
    pytest.importorskip("cv2")
    from pipeline import _activity_gate
    raw = np.zeros((29, 14), dtype=np.float32)   # (0,0) == missing sentinel
    for t in range(29):
        raw[t, 2] = 300 + 8.0 * t                 # only ONE player moving
        raw[t, 3] = 500
    is_dead, gate = _activity_gate(raw)
    assert is_dead
    assert gate["gate_players_present"] < 3


# ---------------------------------------------------------------------------
# label mapping sanity
# ---------------------------------------------------------------------------

def test_unclassified_is_class_zero():
    from label_clips import LABEL_INDEX, LABEL_TO_INDEX
    assert LABEL_INDEX[0] == "Unclassified"
    assert LABEL_TO_INDEX["Unclassified"] == 0


def test_model_has_five_classes():
    pytest.importorskip("torch")  # mamba_model imports torch at module scope
    from mamba_model import LABEL_NAMES, NUM_CLASSES
    assert NUM_CLASSES == 5
    assert "Unclassified" in LABEL_NAMES
