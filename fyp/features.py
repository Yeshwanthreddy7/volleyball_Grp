"""
features.py – Identity-stable, permutation-invariant feature engineering (v2).

This module exists to fix three correctness loopholes that silently corrupted
the original pipeline:

  (L1) IDENTITY SWAP
       In the original training CSVs the columns p1..p6 do NOT track the same
       physical player across frames – the player occupying slot `p_i` changes
       from frame to frame.  Any feature computed as `np.diff(positions[:, i])`
       (velocity, sync_score, "reaction speed", "top-2 fastest") is therefore
       meaningless: it measures the jump from one player to a DIFFERENT player.
       Verified on the shipped data: the median frame-to-frame displacement of
       slot p1 is several hundred cm – physically impossible for a real player
       at 30 FPS (a real player moves ~5–40 cm/frame).

  (L2) (0,0) MISSING SENTINEL
       Untracked players were encoded as the coordinate (0, 0).  (0, 0) is a
       *valid* court location (a corner), so the model cannot distinguish
       "player at the corner" from "player not seen".  Worse, the transition
       real->(0,0)->real injects a huge fake velocity spike.  We use NaN for
       "missing" and provide an explicit presence mask.

  (L3) PERMUTATION SENSITIVITY
       A team formation is a *set* of players, not an ordered tuple.  The model
       should give the same answer regardless of the order players are listed.

The fix has two parts:

  1. `recover_identity()` – greedy / Hungarian nearest-neighbour matching that
     re-threads the per-frame coordinate soup into consistent tracks, so
     velocity is computed player-by-player instead of slot-by-slot.

  2. `team_features()` – a compact, PERMUTATION-INVARIANT descriptor per frame
     (centroid, spread, convex-hull area, nearest-neighbour stats, ball
     relation, count present).  These are order-independent by construction, so
     the identity problem cannot reach the classifier through them.

Everything here is pure NumPy and has zero dependency on torch / YOLO, so it is
unit-testable in isolation (see tests/test_features.py).
"""

from __future__ import annotations

import numpy as np

# Optional SciPy: gives an optimal assignment.  We degrade gracefully to a
# greedy matcher when SciPy is unavailable, so the module never hard-fails.
try:
    from scipy.optimize import linear_sum_assignment  # type: ignore
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False


N_PLAYERS = 6
MISSING = np.nan


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def foot_point(xyxy: np.ndarray) -> np.ndarray:
    """
    Court-contact point of a player bounding box.

    A player touches the court at their FEET, not at the box centre.  Under any
    perspective the bottom-centre of the box is the correct point to map to the
    floor plane; the centroid floats ~half a body-height up-court and injects a
    systematic, distance-dependent position error.  This single change removes
    a bias that grows with how far the player is from the camera.

    Parameters
    ----------
    xyxy : (..., 4) array of [x1, y1, x2, y2] in pixels.

    Returns
    -------
    (..., 2) array of [x_foot, y_foot] = [ (x1+x2)/2 , y2 ].
    """
    xyxy = np.asarray(xyxy, dtype=float)
    x1, y1, x2, y2 = xyxy[..., 0], xyxy[..., 1], xyxy[..., 2], xyxy[..., 3]
    return np.stack([(x1 + x2) / 2.0, y2], axis=-1)


def _pairwise_dist(points: np.ndarray) -> np.ndarray:
    """Full (k, k) Euclidean distance matrix for (k, 2) points."""
    diff = points[:, None, :] - points[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def convex_hull_area(points: np.ndarray) -> float:
    """
    Area of the convex hull of (k, 2) points (monotone chain, no SciPy needed).

    A good scalar proxy for "how much court the formation occupies" – small =
    clustered, large = spread.  Returns 0 for < 3 points.
    """
    pts = np.asarray(points, dtype=float)
    pts = pts[~np.isnan(pts).any(axis=1)]
    if len(pts) < 3:
        return 0.0
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = np.array(lower[:-1] + upper[:-1])
    if len(hull) < 3:
        return 0.0
    x, y = hull[:, 0], hull[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


# ---------------------------------------------------------------------------
# L1 fix – identity recovery
# ---------------------------------------------------------------------------

def recover_identity(
    positions: np.ndarray,
    max_link_dist: float = 250.0,
) -> np.ndarray:
    """
    Re-thread a (T, K, 2) array of per-frame player coordinates into consistent
    tracks, so that output[:, j] is (as far as possible) the SAME player over
    time.

    The shipped CSVs list players in an arbitrary order each frame.  We link
    each frame to the previous one by solving a minimum-cost assignment on the
    distance matrix (optimal via SciPy's Hungarian algorithm, greedy fallback
    otherwise).  Links longer than `max_link_dist` cm are rejected (the player
    probably left / entered tracking) and become NaN.

    Missing inputs may be encoded as NaN or as the legacy (0, 0) sentinel –
    both are treated as "absent" and never linked.

    Parameters
    ----------
    positions    : (T, K, 2) float array. NaN or (0,0) == missing.
    max_link_dist: maximum cm a player may move between frames to be linked.

    Returns
    -------
    (T, K, 2) float array with NaN for missing, identity-consistent columns.
    """
    pos = np.array(positions, dtype=float)
    T, K, _ = pos.shape

    # Normalise the legacy (0,0) sentinel to NaN.
    zero_mask = np.all(pos == 0.0, axis=-1)
    pos[zero_mask] = np.nan

    out = np.full_like(pos, np.nan)
    out[0] = pos[0]

    for t in range(1, T):
        prev = out[t - 1]          # (K, 2) – canonical slots
        cur = pos[t]               # (K, 2) – this frame, arbitrary order

        prev_valid = np.where(~np.isnan(prev).any(axis=1))[0]
        cur_valid = np.where(~np.isnan(cur).any(axis=1))[0]

        if len(prev_valid) == 0 or len(cur_valid) == 0:
            # Nothing to link against – seed slots directly.
            for slot, ci in enumerate(cur_valid):
                out[t, slot] = cur[ci]
            continue

        cost = np.linalg.norm(
            prev[prev_valid][:, None, :] - cur[cur_valid][None, :, :], axis=-1
        )  # (P, C)

        if _HAS_SCIPY:
            rows, cols = linear_sum_assignment(cost)
            pairs = list(zip(rows, cols))
        else:  # greedy fallback
            pairs = []
            used_r, used_c = set(), set()
            order = np.dstack(np.unravel_index(np.argsort(cost, axis=None), cost.shape))[0]
            for r, c in order:
                if r in used_r or c in used_c:
                    continue
                used_r.add(r); used_c.add(c); pairs.append((r, c))

        assigned_cur = set()
        for r, c in pairs:
            if cost[r, c] <= max_link_dist:
                slot = prev_valid[r]
                out[t, slot] = cur[cur_valid[c]]
                assigned_cur.add(cur_valid[c])

        # Place any unmatched current detections into free slots.
        free = [s for s in range(K) if np.isnan(out[t, s]).any()]
        for ci in cur_valid:
            if ci in assigned_cur:
                continue
            if free:
                out[t, free.pop(0)] = cur[ci]

    return out


# ---------------------------------------------------------------------------
# L1 fix – identity-aware kinematics
# ---------------------------------------------------------------------------

def player_velocities(tracked: np.ndarray, fps: float = 30.0) -> np.ndarray:
    """
    Per-player velocity (cm/s) from an identity-consistent (T, K, 2) array.

    Only computed across frames where the SAME slot is present in both t-1 and
    t; gaps yield NaN instead of a fake teleport velocity.  Must be fed the
    output of `recover_identity`, never the raw slot soup.
    """
    tr = np.asarray(tracked, dtype=float)
    disp = np.diff(tr, axis=0)                    # (T-1, K, 2)
    speed = np.linalg.norm(disp, axis=-1) * fps   # (T-1, K) cm/s
    return speed


def sync_score(tracked: np.ndarray) -> float:
    """
    Movement-synchronisation score in [-1, 1] computed on IDENTITY-CONSISTENT
    tracks (the original computed it on the slot soup, so it was noise).

    Mean pairwise cosine similarity of player velocity vectors over all frames
    where both players of a pair are moving and present.
    """
    tr = np.asarray(tracked, dtype=float)
    vel = np.diff(tr, axis=0)                      # (T-1, K, 2)
    scores: list[float] = []
    for v in vel:
        norms = np.linalg.norm(v, axis=1)
        active = np.isfinite(norms) & (norms >= 1e-6)
        if active.sum() < 2:
            continue
        u = v[active] / norms[active][:, None]
        sim = u @ u.T
        k = int(active.sum())
        ri, ci = np.triu_indices(k, k=1)
        scores.extend(sim[ri, ci].tolist())
    return float(np.nanmean(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# Occlusion gap interpolation (spec §2C)
# ---------------------------------------------------------------------------

def interpolate_gaps(
    tracked: np.ndarray,
    max_gap: int = 15,
    kind: str = "linear",
) -> np.ndarray:
    """
    Bridge short tracking gaps caused by occlusion.

    For every track column, interior runs of NaN that are (a) bounded by valid
    detections on BOTH sides and (b) no longer than ``max_gap`` frames are
    filled by 1-D interpolation:

        x(t) = x1 + (x2 - x1) * (t - t1) / (t2 - t1)          (linear)

    or by a cubic spline through the valid samples when ``kind="cubic"`` and
    SciPy is available with >= 4 support points.  Gaps longer than ``max_gap``
    and leading/trailing NaN runs are left untouched: extrapolating a player
    who genuinely left the frame would fabricate data.

    MUST be applied AFTER `recover_identity` – interpolating the raw slot soup
    would bridge between two different physical players.

    Parameters
    ----------
    tracked : (T, K, 2) identity-consistent coordinates, NaN = missing.
    max_gap : longest occlusion run (frames) that will be bridged.
    kind    : "linear" (default) or "cubic".

    Returns
    -------
    (T, K, 2) copy with short gaps filled.
    """
    tr = np.array(tracked, dtype=float)
    T, K, _ = tr.shape
    if T < 3:
        return tr

    use_cubic = kind == "cubic"
    if use_cubic:
        try:
            from scipy.interpolate import CubicSpline  # type: ignore
        except Exception:
            use_cubic = False

    for k in range(K):
        valid = ~np.isnan(tr[:, k]).any(axis=1)
        v_idx = np.where(valid)[0]
        if len(v_idx) < 2:
            continue

        # Identify interior gaps bounded on both sides.
        fill_ts: list[int] = []
        for a, b in zip(v_idx[:-1], v_idx[1:]):
            gap = b - a - 1
            if 0 < gap <= max_gap:
                fill_ts.extend(range(a + 1, b))
        if not fill_ts:
            continue

        ts = np.array(fill_ts)
        for ax in range(2):
            series = tr[:, k, ax]
            if use_cubic and len(v_idx) >= 4:
                cs = CubicSpline(v_idx, series[v_idx])
                tr[ts, k, ax] = cs(ts)
            else:
                tr[ts, k, ax] = np.interp(ts, v_idx, series[v_idx])

    return tr


# ---------------------------------------------------------------------------
# Kinematic per-frame features (spec §1B: speed differentials + sync)
# ---------------------------------------------------------------------------

KINEMATIC_COLS = [
    "speed_top2",   # mean speed of the 2 fastest players this frame (cm/frame)
    "speed_bot4",   # mean speed of the remaining (<=4) players    (cm/frame)
    "speed_diff",   # top2 - bot4: separates frontline attackers from base
    "sync_inst",    # instantaneous synchronisation (mean pairwise cos-sim)
]


def kinematic_features(tracked: np.ndarray) -> np.ndarray:
    """
    Per-frame kinematic descriptors from an IDENTITY-CONSISTENT (T, K, 2) array.

    Frame t uses the displacement t-1 -> t; frame 0 is zeros.  All values are
    permutation-invariant (speeds are sorted; sync is a mean over pairs).

    Returns
    -------
    (T, len(KINEMATIC_COLS)) float32 array, NaN-free.
    """
    tr = np.asarray(tracked, dtype=float)
    T = len(tr)
    out = np.zeros((T, len(KINEMATIC_COLS)), dtype=np.float32)
    if T < 2:
        return out

    vel = np.diff(tr, axis=0)                      # (T-1, K, 2)
    for t in range(1, T):
        v = vel[t - 1]
        speed = np.linalg.norm(v, axis=1)          # (K,)
        finite = np.isfinite(speed)
        s = np.sort(speed[finite])[::-1]
        if len(s) > 0:
            top2 = float(s[: min(2, len(s))].mean())
            rest = s[2:6]
            bot4 = float(rest.mean()) if len(rest) else 0.0
            out[t, 0] = top2
            out[t, 1] = bot4
            out[t, 2] = top2 - bot4

        moving = finite & (speed >= 1e-6)
        if moving.sum() >= 2:
            u = v[moving] / speed[moving][:, None]
            sim = u @ u.T
            k = int(moving.sum())
            ri, ci = np.triu_indices(k, k=1)
            out[t, 3] = float(sim[ri, ci].mean())

    return out


# ---------------------------------------------------------------------------
# L3 fix – permutation-invariant per-frame descriptor
# ---------------------------------------------------------------------------

PERM_INVARIANT_COLS = [
    "ball_present",
    "ball_x", "ball_y",
    "centroid_x", "centroid_y",
    "spread_x", "spread_y",
    "hull_area",
    "nn_dist_mean", "nn_dist_min", "nn_dist_max",
    "max_pair_dist",
    "n_present",
    "ball_to_centroid",
]


def team_features(players_frame: np.ndarray, ball_xy: np.ndarray) -> np.ndarray:
    """
    Build a permutation-invariant descriptor for ONE frame.

    Parameters
    ----------
    players_frame : (K, 2) player court coords for this frame (NaN = absent).
    ball_xy       : (2,) ball court coords, NaN if ball not detected.

    Returns
    -------
    (len(PERM_INVARIANT_COLS),) float vector – order-independent w.r.t. players.
    """
    pf = np.asarray(players_frame, dtype=float)
    present = pf[~np.isnan(pf).any(axis=1)]
    n = len(present)

    ball = np.asarray(ball_xy, dtype=float)
    ball_present = float(np.all(np.isfinite(ball)))
    bx, by = (ball if ball_present else np.array([0.0, 0.0]))

    if n == 0:
        centroid = np.array([0.0, 0.0]); spread = np.array([0.0, 0.0])
        hull = 0.0; nn_mean = nn_min = nn_max = max_pair = 0.0
        ball_cen = 0.0
    else:
        centroid = present.mean(axis=0)
        spread = present.std(axis=0) if n > 1 else np.array([0.0, 0.0])
        hull = convex_hull_area(present)
        if n >= 2:
            d = _pairwise_dist(present)
            np.fill_diagonal(d, np.inf)
            nn = d.min(axis=1)
            nn_mean, nn_min = float(nn.mean()), float(nn.min())
            finite = d[np.isfinite(d)]
            nn_max = float(nn.max())
            max_pair = float(finite.max()) if finite.size else 0.0
        else:
            nn_mean = nn_min = nn_max = max_pair = 0.0
        ball_cen = float(np.linalg.norm(centroid - np.array([bx, by]))) if ball_present else 0.0

    return np.array([
        ball_present, bx, by,
        centroid[0], centroid[1],
        spread[0], spread[1],
        hull,
        nn_mean, nn_min, nn_max,
        max_pair,
        float(n),
        ball_cen,
    ], dtype=np.float32)


def sequence_to_perm_invariant(
    players: np.ndarray,   # (T, K, 2)
    ball: np.ndarray,      # (T, 2)
) -> np.ndarray:
    """Stack `team_features` over a clip → (T, len(PERM_INVARIANT_COLS))."""
    players = np.asarray(players, dtype=float)
    ball = np.asarray(ball, dtype=float)
    return np.stack(
        [team_features(players[t], ball[t]) for t in range(len(players))],
        axis=0,
    )


# ---------------------------------------------------------------------------
# Single source of truth: raw clip CSV  ->  model input
# ---------------------------------------------------------------------------

# Raw columns as stored in the clip CSVs (slot-ordered, possibly identity-swapped).
RAW_BALL_COLS = ["ball_x", "ball_y"]
RAW_PLAYER_COLS = [f"p{i}_{ax}" for i in range(1, N_PLAYERS + 1) for ax in ("x", "y")]
RAW_COLS = RAW_BALL_COLS + RAW_PLAYER_COLS  # 14 raw columns


# Plausible coordinate envelope (cm).  The court is 1800x900; we allow a
# generous margin so legitimate out-of-court positions (a player digging wide,
# a ball arcing above the net plane) survive, while the divergent
# constant-velocity ball extrapolations seen in the raw data (ball_x down to
# -14105, ball_y up to +11671) are correctly rejected as "not a real
# measurement".
COURT_W_CM = 1800.0
COURT_H_CM = 900.0
COORD_MARGIN_CM = 900.0
X_LO, X_HI = -COORD_MARGIN_CM, COURT_W_CM + COORD_MARGIN_CM
Y_LO, Y_HI = -COORD_MARGIN_CM - COURT_H_CM, COURT_H_CM + COORD_MARGIN_CM


def _sanitize(arr: np.ndarray) -> np.ndarray:
    """Set coordinates outside the plausible court envelope to NaN (in place)."""
    x, y = arr[..., 0], arr[..., 1]
    bad = (x < X_LO) | (x > X_HI) | (y < Y_LO) | (y > Y_HI)
    arr[bad] = np.nan
    return arr


def raw_frame_to_arrays(seq_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Split a (T, 14) raw feature block into (players (T,K,2), ball (T,2)).

    Cleaning applied:
      • legacy (0,0) sentinels  -> NaN
      • out-of-envelope coords  -> NaN  (kills divergent ball extrapolations)
    """
    seq_raw = np.asarray(seq_raw, dtype=float)
    ball = seq_raw[:, 0:2].copy()
    players = seq_raw[:, 2:].reshape(len(seq_raw), N_PLAYERS, 2).copy()
    players[np.all(players == 0.0, axis=-1)] = np.nan
    ball[np.all(ball == 0.0, axis=-1)] = np.nan
    _sanitize(players)
    _sanitize(ball)
    return players, ball


# The model-input contract.  Bump FEATURE_VERSION whenever the semantics of
# the produced tensor change: checkpoint loaders compare this tag and refuse
# to serve silently-mismatched features.
MODEL_FEATURE_COLS = PERM_INVARIANT_COLS + KINEMATIC_COLS
MODEL_INPUT_DIM = len(MODEL_FEATURE_COLS)   # 18
FEATURE_VERSION = "perm_invariant_v2"
INTERP_MAX_GAP = 15                          # spec §2C occlusion bridge limit


def build_model_sequence(seq_raw: np.ndarray, target_len: int = 29) -> np.ndarray:
    """
    Convert a raw (T, 14) slot-ordered clip into the corrected, identity-aware,
    permutation-invariant model input of shape (target_len, MODEL_INPUT_DIM).

    Pipeline:  raw soup  ->  recover_identity        (fixes L1: identity swap)
                          ->  interpolate_gaps       (bridges <=15-frame occlusion)
                          ->  perm-invariant descriptor (fixes L2 missing & L3 order)
                          ->  + kinematic features   (speed top2/bot4/diff, sync)
                          ->  pad / truncate to target_len.

    This is the ONE function used by training, inference and the live video
    pipeline, guaranteeing train/serve representation parity.
    """
    players, ball = raw_frame_to_arrays(seq_raw)
    tracked = recover_identity(players)
    tracked = interpolate_gaps(tracked, max_gap=INTERP_MAX_GAP)
    ball = interpolate_gaps(ball[:, None, :], max_gap=INTERP_MAX_GAP)[:, 0, :]

    feats = sequence_to_perm_invariant(tracked, ball)   # (T, 14)
    kin = kinematic_features(tracked)                   # (T, 4)
    feats = np.concatenate([feats, kin], axis=1)        # (T, 18)

    if len(feats) < target_len:
        pad = np.zeros((target_len - len(feats), feats.shape[1]), dtype=np.float32)
        feats = np.vstack([feats, pad])
    else:
        feats = feats[:target_len]
    return feats.astype(np.float32)


__all__ = [
    "N_PLAYERS", "MISSING", "PERM_INVARIANT_COLS", "KINEMATIC_COLS",
    "MODEL_FEATURE_COLS", "MODEL_INPUT_DIM", "FEATURE_VERSION",
    "INTERP_MAX_GAP", "RAW_COLS",
    "foot_point", "convex_hull_area",
    "recover_identity", "interpolate_gaps", "kinematic_features",
    "player_velocities", "sync_score",
    "team_features", "sequence_to_perm_invariant",
    "raw_frame_to_arrays", "build_model_sequence",
]
