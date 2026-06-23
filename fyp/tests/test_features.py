"""
test_features.py - proves the correctness fixes in features.py actually work.

Run:  python -m pytest fyp/tests/test_features.py -v
  or: python fyp/tests/test_features.py        (no pytest needed)

Dependency-light (NumPy only) so it runs on any machine without torch /
ultralytics / a GPU.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features import (  # noqa: E402
    foot_point,
    convex_hull_area,
    recover_identity,
    player_velocities,
    sync_score,
    team_features,
    sequence_to_perm_invariant,
    raw_frame_to_arrays,
    build_model_sequence,
    PERM_INVARIANT_COLS,
)


def test_foot_point():
    box = np.array([100, 200, 140, 360])  # x1,y1,x2,y2
    fp = foot_point(box)
    assert fp[0] == 120 and fp[1] == 360  # bottom-centre, not centroid (y=280)


def test_identity_recovery_fixes_swap():
    """Two players that SWAP slots every frame must be re-threaded so each
    column moves smoothly instead of teleporting."""
    A = np.array([100.0, 100.0])
    B = np.array([800.0, 100.0])
    T = 10
    raw = np.full((T, 2, 2), np.nan)
    for t in range(T):
        a = A + np.array([t * 5.0, 0.0])
        b = B + np.array([t * 5.0, 0.0])
        if t % 2 == 0:
            raw[t, 0], raw[t, 1] = a, b
        else:
            raw[t, 0], raw[t, 1] = b, a  # swap order

    raw_jump = np.nanmedian(np.linalg.norm(np.diff(raw[:, 0], axis=0), axis=1))
    assert raw_jump > 500
    fixed = recover_identity(raw, max_link_dist=250)
    fixed_jump = np.nanmedian(np.linalg.norm(np.diff(fixed[:, 0], axis=0), axis=1))
    assert fixed_jump < 50, "identity not recovered (jump=%.1f)" % fixed_jump


def test_zero_sentinel_treated_as_missing():
    raw = np.array([[[0.0, 0.0], [500.0, 500.0]]])
    fixed = recover_identity(raw)
    assert np.isnan(fixed[0, 0]).all()
    assert np.allclose(fixed[0, 1], [500, 500])


def test_velocity_ignores_gaps():
    tr = np.array([[[0.0, 0.0]], [[np.nan, np.nan]], [[10.0, 0.0]]])
    sp = player_velocities(tr, fps=30.0)
    assert np.isnan(sp[0, 0]) or np.isnan(sp[1, 0])


def test_perm_invariance():
    players = np.array([[100, 100], [400, 200], [700, 300], [200, 600]], float)
    ball = np.array([500.0, 450.0])
    base = team_features(players, ball)
    rng = np.random.default_rng(0)
    for _ in range(20):
        perm = rng.permutation(len(players))
        assert np.allclose(team_features(players[perm], ball), base, equal_nan=True)


def test_convex_hull_area_square():
    sq = np.array([[0, 0], [0, 10], [10, 10], [10, 0]], float)
    assert abs(convex_hull_area(sq) - 100.0) < 1e-6


def test_sync_score_bounds_and_meaning():
    base = np.array([[0, 0], [100, 0], [200, 0]], float)
    seq = np.stack([base + np.array([d * 10, 0]) for d in range(5)])
    assert sync_score(seq) > 0.99
    seq2 = np.stack(
        [base + np.array([[d * 10, 0], [-d * 10, 0], [d * 10, 0]]) for d in range(5)]
    )
    assert sync_score(seq2) < 0.5


def test_sequence_shape():
    T, K = 29, 6
    players = np.random.default_rng(1).normal(500, 100, size=(T, K, 2))
    ball = np.random.default_rng(2).normal(500, 100, size=(T, 2))
    feats = sequence_to_perm_invariant(players, ball)
    assert feats.shape == (T, len(PERM_INVARIANT_COLS))
    assert np.isfinite(feats).all()


def test_divergent_ball_rejected():
    """A divergent constant-velocity ball extrapolation (real data: ball_x ~
    -14000) must be treated as missing, not fed to the model."""
    raw = np.zeros((29, 14), dtype=np.float32)
    raw[:, 0] = -14105.0
    raw[:, 1] = 11671.0
    raw[:, 2:4] = [500.0, 500.0]
    _, ball = raw_frame_to_arrays(raw)
    assert np.isnan(ball).all()
    seq = build_model_sequence(raw)
    assert np.isfinite(seq).all()
    assert (seq[:, 0] == 0).all()  # ball_present == 0 every frame




def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print("  PASS  " + fn.__name__)
    print("\n{}/{} tests passed.".format(len(fns), len(fns)))


if __name__ == "__main__":
    _run_all()
