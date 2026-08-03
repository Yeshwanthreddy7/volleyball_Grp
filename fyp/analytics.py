"""
analytics.py - data-mapped tactical & biomechanical analytics for the volleyball
project.

Everything here is computed from the ACTUAL clip CSVs through the corrected,
identity-aware feature pipeline (features.py). Each output number is therefore
traceable to data, and each is mapped to a sourced biomechanics reference range
so an expert can see "measured-vs-literature" side by side.

It is pure NumPy / pandas (no torch / YOLO), so it runs anywhere and is fully
unit-tested. Three capabilities, all from the feasible/defensible tier:

  1. clip_metrics()        - measured tactical proxies per clip (cm, cm/s, etc.)
  2. tactical_deviation    - unsupervised anomaly score: how far a clip sits from
                             the "coordinated" reference distribution (Mahalanobis).
                             Needs no labels - answers "intelligence in deviations".
  3. confidence_label()    - confidence-based abstention ("Uncertain") so the
                             system never forces a label it cannot support.

Plus BIOMETRIC_REFERENCE: literature anchors (PubMed/journals) used to interpret
the measured numbers. Distances are in court units (~cm under the project's
1800x900 linear court scaling); they become metric cm only with a calibrated
homography - stated honestly, no overclaim.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, asdict

import warnings

import numpy as np
import pandas as pd

from features import (
    RAW_COLS,
    raw_frame_to_arrays,
    recover_identity,
    player_velocities,
    sync_score,
    convex_hull_area,
)

FPS = 30.0
# Physiological cap: a player cannot move faster than ~15 m/s; larger
# frame-to-frame jumps are tracking noise, not motion, and are dropped so
# they never inflate a velocity average.
MAX_DISP_CM_PER_FRAME = 50.0
# Tight identity-link gate (cm/frame). 120 cm/frame ~ 3.6 m/s is already a
# fast court move; beyond this, a 'link' is almost certainly two different
# players, so we refuse it to avoid teleport velocities.
ID_LINK_CM = 120.0


# ---------------------------------------------------------------------------
# Biometric / biomechanical reference (sourced) used to interpret measurements.
# Each entry: (low, high, unit, what it anchors, citation).
# ---------------------------------------------------------------------------
BIOMETRIC_REFERENCE = {
    "attacker_takeoff_velocity": (
        260, 300, "cm/s",
        "Elite approach take-off velocity ~2.9 m/s; anchors top-player speed in a "
        "Coordinated Attack.",
        "Kinematic analysis of the volleyball attack, J. Human Kinetics 2017 (PMC5548173)",
    ),
    "defender_reaction_time": (
        650, 770, "ms",
        "Whole-body/upper-extremity reaction time; anchors the 'late support' lag.",
        "Reaction time by playing position, The Sport Journal",
    ),
    "spike_reach_elite_men": (
        345, 360, "cm",
        "Elite men spike reach ~354 cm (context only; NOT measured by this system).",
        "Influence of jump height on game efficiency, Sci. Reports 2023 (PMC10235019)",
    ),
    "cmj_middle_blocker": (
        30, 42, "cm",
        "Middle-blocker counter-movement jump ~36 cm (context for vertical actions).",
        "Anthropometric & vertical-jump abilities by position, IJERPH 2021 (PMC8393901)",
    ),
}

# Reaction-time reference expressed in frames at 30 FPS, for the lag metric.
REACTION_LAG_REF_FRAMES = (round(650 / 1000 * FPS), round(770 / 1000 * FPS))  # ~(20, 23)


# ---------------------------------------------------------------------------
# Per-clip measured metrics
# ---------------------------------------------------------------------------

METRIC_COLS = [
    "n_frames", "ball_present_frac",
    "top2_speed_cms", "rest_speed_cms", "speed_ratio",
    "mean_spacing_cm", "centroid_vel_cms", "sync_score",
    "mean_hull_area", "reaction_lag_frames",
]


def _per_player_mean_speed(tracked: np.ndarray) -> np.ndarray:
    """Mean speed (cm/s) per player slot over the clip, NaN-safe (identity-aware).

    Per-frame displacements above the physiological cap are treated as tracking
    noise (set to NaN) so a single bad frame cannot inflate the average.
    """
    disp = np.linalg.norm(np.diff(tracked, axis=0), axis=-1)   # (T-1, K) cm/frame
    disp = np.where(disp > MAX_DISP_CM_PER_FRAME, np.nan, disp)
    sp = disp * FPS                                            # cm/s
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_sp = np.nanmean(sp, axis=0)                       # (K,)
    return mean_sp


def _reaction_lag_frames(tracked: np.ndarray, ball: np.ndarray) -> float:
    """Frames between ball impact and the nearest player's peak-speed frame.

    Impact = first frame where ball vertical velocity reverses sign. Returns NaN
    when the ball is absent (so it never fabricates a value).
    """
    if not np.isfinite(ball).all(axis=1).any():
        return float("nan")
    by = ball[:, 1]
    valid = np.isfinite(by)
    if valid.sum() < 3:
        return float("nan")
    vy = np.diff(by)
    impact = None
    for i in range(1, len(vy)):
        if np.isfinite(vy[i - 1]) and np.isfinite(vy[i]) and vy[i - 1] * vy[i] <= 0:
            impact = i
            break
    if impact is None:
        return float("nan")
    # ball position at impact
    bpt = ball[impact]
    if not np.isfinite(bpt).all():
        return float("nan")
    frame = tracked[impact]
    d = np.linalg.norm(frame - bpt, axis=1)
    if not np.isfinite(d).any():
        return float("nan")
    closest = int(np.nanargmin(d))
    traj = tracked[:, closest, :]
    speed = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    if not np.isfinite(speed).any():
        return float("nan")
    peak = int(np.nanargmax(speed))
    return float(peak - impact)


def clip_metrics(raw: np.ndarray) -> dict:
    """Measured tactical/biomechanical proxies for one raw (T,14) clip."""
    players, ball = raw_frame_to_arrays(raw)
    tracked = recover_identity(players, max_link_dist=ID_LINK_CM)
    T = len(tracked)

    mean_sp = _per_player_mean_speed(tracked)
    finite = mean_sp[np.isfinite(mean_sp)]
    if finite.size >= 2:
        order = np.sort(finite)[::-1]
        top2 = float(order[:2].mean())
        rest = float(order[2:].mean()) if order.size > 2 else float(order[:2].mean())
    elif finite.size == 1:
        top2 = rest = float(finite[0])
    else:
        top2 = rest = 0.0
    speed_ratio = float(top2 / rest) if rest > 1e-6 else 0.0

    # spacing + centroid velocity (present players only)
    spac = []
    cents = []
    for f in tracked:
        occ = f[~np.isnan(f).any(axis=1)]
        if len(occ) >= 2:
            dd = np.linalg.norm(occ[:, None, :] - occ[None, :, :], axis=-1)
            np.fill_diagonal(dd, np.inf)
            spac.extend(dd.min(axis=1).tolist())
        cents.append(occ.mean(axis=0) if len(occ) else [np.nan, np.nan])
    mean_spacing = float(np.mean(spac)) if spac else 0.0
    cents = np.array(cents)
    cvel = np.linalg.norm(np.diff(cents, axis=0), axis=1)
    cvel = np.where(cvel > MAX_DISP_CM_PER_FRAME, np.nan, cvel)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        centroid_vel = float(np.nanmean(cvel) * FPS) if np.isfinite(cvel).any() else 0.0

    hulls = [convex_hull_area(f[~np.isnan(f).any(axis=1)]) for f in tracked]
    mean_hull = float(np.mean(hulls)) if hulls else 0.0

    return {
        "n_frames": T,
        "ball_present_frac": float(np.isfinite(ball).all(axis=1).mean()),
        "top2_speed_cms": round(top2, 1),
        "rest_speed_cms": round(rest, 1),
        "speed_ratio": round(speed_ratio, 2),
        "mean_spacing_cm": round(mean_spacing, 1),
        "centroid_vel_cms": round(centroid_vel, 1),
        "sync_score": round(sync_score(tracked), 3),
        "mean_hull_area": round(mean_hull, 0),
        "reaction_lag_frames": _reaction_lag_frames(tracked, ball),
    }


# ---------------------------------------------------------------------------
# Unsupervised tactical-deviation score (Mahalanobis from coordinated reference)
# ---------------------------------------------------------------------------

DEVIATION_FEATURES = [
    "top2_speed_cms", "rest_speed_cms", "speed_ratio",
    "mean_spacing_cm", "centroid_vel_cms", "sync_score", "mean_hull_area",
]


@dataclass
class DeviationModel:
    """Unsupervised tactical-novelty model.

    Features are z-standardised (population mean/std) before a Mahalanobis
    distance is taken from the 'coordinated' reference cluster, so no single
    large-magnitude feature (e.g. hull area) dominates. Higher score = more
    kinematically unusual relative to coordinated play.

    Honest scope: this is an exploratory NOVELTY score, not a clean
    coordinated-vs-breakdown classifier. On the project data it ranks Delayed
    Support highest and Coordinated Defense lowest; Coordinated Attack also
    scores high because explosive attacking motion is itself kinematically
    extreme. Reported as a continuous indicator, never as a hard label.
    """
    feat_mean: np.ndarray   # population mean (for standardisation)
    feat_std: np.ndarray    # population std  (for standardisation)
    ref_mean: np.ndarray    # reference-cluster mean in standardised space
    inv_cov: np.ndarray
    cols: list
    col_median: np.ndarray  # for NaN imputation at scoring time

    def score(self, metrics: dict) -> float:
        x = np.array([metrics.get(c, np.nan) for c in self.cols], dtype=float)
        x = np.where(np.isfinite(x), x, self.col_median)
        z = (x - self.feat_mean) / self.feat_std
        d = z - self.ref_mean
        return float(np.sqrt(max(d @ self.inv_cov @ d, 0.0)))


def fit_deviation_model(metrics_df: pd.DataFrame, reference_mask: np.ndarray,
                        cols=None) -> DeviationModel:
    """Fit the standardised-Mahalanobis novelty model.

    `reference_mask` selects the 'coordinated' clips (the normal pattern); the
    score is each clip's standardised Mahalanobis distance from that cluster.
    Needs no labels beyond choosing the reference set.
    """
    cols = cols or DEVIATION_FEATURES
    X = metrics_df[cols].to_numpy(float)
    col_median = np.nanmedian(X, axis=0)
    X = np.where(np.isfinite(X), X, col_median)
    feat_mean = X.mean(axis=0)
    feat_std = X.std(axis=0) + 1e-9
    Z = (X - feat_mean) / feat_std
    ref = Z[reference_mask]
    ref_mean = ref.mean(axis=0)
    cov = np.cov(ref, rowvar=False)
    cov += np.eye(cov.shape[0]) * 1e-2          # regularise (near-singular dirs)
    inv = np.linalg.pinv(cov)
    return DeviationModel(feat_mean=feat_mean, feat_std=feat_std, ref_mean=ref_mean,
                          inv_cov=inv, cols=list(cols), col_median=col_median)


# ---------------------------------------------------------------------------
# Confidence-based abstention (Unclassified handling, no retrain)
# ---------------------------------------------------------------------------

def confidence_label(probs, label_names, tau: float = 0.5) -> tuple:
    """Return (label, confidence). If max prob < tau, label = 'Uncertain'."""
    probs = np.asarray(probs, dtype=float)
    i = int(np.argmax(probs))
    conf = float(probs[i])
    return (label_names[i] if conf >= tau else "Uncertain", conf)


# ---------------------------------------------------------------------------
# Directory analysis  ->  data-mapped report + CSV
# ---------------------------------------------------------------------------

def analyze_directory(directory: str, out_csv: str | None = None) -> pd.DataFrame:
    """Compute clip_metrics for every CSV, add a tactical-deviation score, and
    return a per-clip DataFrame mapped to data. Writes out_csv if given."""
    files = sorted(glob.glob(os.path.join(directory, "*.csv")))
    rows = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if not set(RAW_COLS) <= set(df.columns):
            continue
        raw = df[RAW_COLS].to_numpy(float)
        m = clip_metrics(raw)
        m["file"] = os.path.basename(f)
        m["label"] = df["target_label"].iloc[0] if "target_label" in df.columns else ""
        rows.append(m)
    out = pd.DataFrame(rows)
    if len(out):
        ref_mask = out["label"].isin(["Coordinated Attack", "Coordinated Defense"]).to_numpy()
        if ref_mask.sum() >= len(DEVIATION_FEATURES) + 1:
            model = fit_deviation_model(out, ref_mask)
            out["tactical_deviation"] = [model.score(r) for r in out.to_dict("records")]
    if out_csv and len(out):
        out.to_csv(out_csv, index=False)
    return out


def reference_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Map measured population values to the sourced biometric reference ranges."""
    pop = {
        "attacker_takeoff_velocity": metrics_df["top2_speed_cms"].median(),
        "defender_reaction_time": metrics_df["reaction_lag_frames"].median(),
    }
    recs = []
    for key, (lo, hi, unit, note, cite) in BIOMETRIC_REFERENCE.items():
        measured = pop.get(key, None)
        if key == "defender_reaction_time" and measured is not None and np.isfinite(measured):
            measured_disp = f"{measured:.0f} frames (~{measured/FPS*1000:.0f} ms)"
            ref_disp = f"{lo}-{hi} ms (~{REACTION_LAG_REF_FRAMES[0]}-{REACTION_LAG_REF_FRAMES[1]} frames)"
        elif measured is not None and np.isfinite(measured):
            measured_disp = f"{measured:.0f} {unit}"
            ref_disp = f"{lo}-{hi} {unit}"
        else:
            measured_disp = "not measured by this system"
            ref_disp = f"{lo}-{hi} {unit}"
        recs.append({
            "biometric_anchor": key,
            "measured_population": measured_disp,
            "literature_reference": ref_disp,
            "note": note,
            "source": cite,
        })
    return pd.DataFrame(recs)


if __name__ == "__main__":
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else "training_csv"
    out_csv = sys.argv[2] if len(sys.argv) > 2 else "coordination_analysis.csv"
    df = analyze_directory(d, out_csv=out_csv)
    print("=" * 70)
    print(f"COORDINATION ANALYSIS  -  {len(df)} clips from '{d}'")
    print("=" * 70)
    if "tactical_deviation" in df.columns:
        print("\nMean measured metrics + tactical-deviation score, BY CLASS:")
        summ = df.groupby("label")[
            ["top2_speed_cms", "rest_speed_cms", "speed_ratio", "mean_spacing_cm",
             "centroid_vel_cms", "sync_score", "tactical_deviation"]
        ].mean().round(2)
        print(summ.to_string())
        print("\n(Deviation should be LOWEST for the coordinated classes - validation"
              " that the unsupervised score is meaningful.)")
    print("\nMEASURED-vs-LITERATURE biometric mapping:")
    print(reference_table(df).to_string(index=False))
    print(f"\nPer-clip data written to '{out_csv}'.")
