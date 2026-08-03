"""
roles.py - behavioural role inference (attacker / setter / libero / defender)
from identity-consistent player trajectories. Numpy-only, fully unit-tested.

What this is (and honestly is not)
----------------------------------
Broadcast footage carries no roster: no jersey-number OCR, no persistent
identity across clips. Roles therefore cannot be *recognised*; they can be
*inferred behaviourally* within each 1-second window from motion signatures
that are well documented in the volleyball literature:

  attacker : explosive approach TOWARD the net - elite spike approaches reach
             ~2.6-3.0 m/s horizontal velocity (Fuchs et al., J. Human
             Kinetics 2017, PMC5548173) - ending in/near the attack zone.
  setter   : operates near the net with LOW total displacement (position-
             holding between contacts; positional jump-load profiles:
             PMC11669427).
  libero   : back-row specialist - never attacks at the net (FIVB rule),
             covers court laterally; distinguished by back-row residency +
             the highest lateral speed share (reaction/defence literature:
             The Sport Journal, coinciding-anticipation timing by position).
  defender : back-row base player without the libero's lateral dominance.

Every label is a SCORED ESTIMATE with an abstain path ("unknown") when the
window carries too little evidence (short visibility, all-static play).
This is stated in the report as an estimate - not an identity claim.

Court convention (matches court.py): x in [0,1800], y in [0,900], net at
y = 450. team_side='bottom' plays y > 450 (back line y=900), 'top' mirrored.
Under linear fallback scaling units are "court units ~ cm"; with a calibrated
homography they are metric cm - same honesty note as analytics.py.
"""
from __future__ import annotations

import numpy as np

NET_Y = 450.0
SIDE_DEPTH = 450.0
ATTACK_ZONE_CM = 150.0          # front zone: within 1.5 m of the net (~3m line scaled)
NEAR_NET_CM = 160.0             # setter operating band
BACK_ROW_CM = 240.0             # clearly behind the attack zone
PHYSIO_SPEED_CAP = 800.0        # cm/s; drop tracking-glitch speeds above this
MIN_VALID_FRAC = 0.4            # abstain below this visibility
MIN_MARGIN = 0.10               # abstain if top-2 role scores this close

ROLES = ("attacker", "setter", "libero", "defender")
UNKNOWN = "unknown"

# Literature anchors used by role_skill_table (cm/s unless noted).
ROLE_LITERATURE = {
    "attacker_approach": (260.0, 300.0, "elite spike-approach velocity, PMC5548173"),
    "libero_lateral":    (150.0, 400.0, "defensive lateral coverage band; position "
                                        "reaction-time literature (Sport Journal)"),
    "setter_displacement": (0.0, 250.0, "position-holding between contacts, "
                                        "PMC11669427 (low locomotor load)"),
}


def _net_dist(y: np.ndarray, team_side: str) -> np.ndarray:
    """Signed distance from the net INTO our side (>=0 on our side)."""
    return (y - NET_Y) if team_side == "bottom" else (NET_Y - y)


def slot_kinematics(tracked: np.ndarray, fps: float = 30.0,
                    team_side: str = "bottom") -> list[dict]:
    """Per-slot motion signature from an identity-consistent (T, K, 2) array.

    NaN = player not visible that frame (handled, not imputed here - feed the
    output of features.recover_identity + interpolate_gaps for best results).
    """
    tracked = np.asarray(tracked, dtype=float)
    T, K, _ = tracked.shape
    out = []
    for k in range(K):
        p = tracked[:, k, :]
        valid = ~np.isnan(p).any(axis=1)
        n_valid = int(valid.sum())
        rec: dict = {"slot": k, "valid_frac": n_valid / max(T, 1)}
        if n_valid < 2:
            rec.update(net_dist_mean=np.nan, net_dist_min=np.nan, vmax=0.0,
                       approach_speed=0.0, lateral_speed=0.0,
                       lateral_ratio=0.0, displacement=0.0)
            out.append(rec)
            continue

        nd = _net_dist(p[:, 1], team_side)
        rec["net_dist_mean"] = float(np.nanmean(np.where(valid, nd, np.nan)))
        rec["net_dist_min"] = float(np.nanmin(np.where(valid, nd, np.nan)))

        # frame-to-frame velocities only across consecutive valid frames
        dv = p[1:] - p[:-1]
        ok = valid[1:] & valid[:-1]
        vx = np.where(ok, dv[:, 0], np.nan) * fps
        vy = np.where(ok, dv[:, 1], np.nan) * fps
        speed = np.hypot(vx, vy)
        glitch = speed > PHYSIO_SPEED_CAP                          # glitch guard
        vx = np.where(glitch, np.nan, vx)
        vy = np.where(glitch, np.nan, vy)
        speed = np.where(glitch, np.nan, speed)

        toward_net = (-vy) if team_side == "bottom" else vy       # cm/s to net
        toward_net = np.where(np.abs(toward_net) > PHYSIO_SPEED_CAP,
                              np.nan, toward_net)

        def q95(a):
            a = a[np.isfinite(a)]
            return float(np.quantile(a, 0.95)) if a.size else 0.0

        rec["vmax"] = q95(speed)
        rec["approach_speed"] = q95(np.clip(toward_net, 0, None))
        rec["lateral_speed"] = q95(np.abs(vx))
        adx = np.nansum(np.abs(np.where(ok, dv[:, 0], np.nan)))
        ady = np.nansum(np.abs(np.where(ok, dv[:, 1], np.nan)))
        rec["lateral_ratio"] = float(adx / (adx + ady)) if (adx + ady) > 0 else 0.0
        step = np.hypot(dv[:, 0], dv[:, 1])
        rec["displacement"] = float(np.nansum(np.where(ok & ~glitch, step, np.nan)))
        out.append(rec)
    return out


def _norm(values: np.ndarray) -> np.ndarray:
    """Scale to [0,1] across slots (max-normalisation, NaN-safe)."""
    v = np.asarray(values, dtype=float)
    m = np.nanmax(v) if np.isfinite(v).any() else 0.0
    return np.where(np.isfinite(v), v / m, 0.0) if m > 0 else np.zeros_like(v)


def infer_roles(tracked: np.ndarray, fps: float = 30.0,
                team_side: str = "bottom") -> list[dict]:
    """Score each slot for the four roles and hard-label with abstention.

    Constraints applied (volleyball structure): at most ONE setter and ONE
    libero per window - competing slots keep the highest score, the rest fall
    back to their next-best role.
    """
    kin = slot_kinematics(tracked, fps=fps, team_side=team_side)
    n = len(kin)
    approach = _norm([r["approach_speed"] for r in kin])
    lateral = _norm([r["lateral_speed"] for r in kin])
    displ = _norm([r["displacement"] for r in kin])

    scores = np.zeros((n, len(ROLES)), dtype=float)
    for i, r in enumerate(kin):
        if r["valid_frac"] < MIN_VALID_FRAC or not np.isfinite(r["net_dist_mean"]):
            continue
        reaches_front = r["net_dist_min"] <= ATTACK_ZONE_CM
        near_net = r["net_dist_mean"] <= NEAR_NET_CM
        back_row = r["net_dist_mean"] >= BACK_ROW_CM
        # attacker: fast approach that actually arrives at the attack zone
        scores[i, 0] = approach[i] * (1.0 if reaches_front else 0.4)
        # setter: lives at the net, moves little
        scores[i, 1] = (1.0 if near_net else 0.2) * (1.0 - 0.8 * displ[i])
        # libero: back row + lateral dominance
        scores[i, 2] = (1.0 if back_row else 0.1) * lateral[i] * \
            (0.5 + 0.5 * r["lateral_ratio"])
        # defender: back row baseline, not sprinting at the net
        scores[i, 3] = (0.7 if back_row else 0.2) * (1.0 - 0.6 * approach[i])

    # exclusivity: one setter, one libero
    for role_idx in (1, 2):
        col = scores[:, role_idx]
        if (col > 0).sum() > 1:
            keep = int(np.argmax(col))
            for i in range(n):
                if i != keep:
                    scores[i, role_idx] = 0.0

    out = []
    for i, r in enumerate(kin):
        s = scores[i]
        order = np.argsort(s)[::-1]
        top, second = s[order[0]], s[order[1]]
        if r["valid_frac"] < MIN_VALID_FRAC or top <= 0 or (top - second) < MIN_MARGIN:
            label = UNKNOWN
        else:
            label = ROLES[int(order[0])]
        rec = dict(r)
        rec["role"] = label
        rec["role_scores"] = {ROLES[j]: round(float(s[j]), 3) for j in range(4)}
        out.append(rec)
    return out


def role_skill_summary(role_records: list[dict]) -> dict:
    """Aggregate per-role skill metrics for one clip -> flat dict for CSV."""
    agg: dict = {}
    for role in ROLES:
        rs = [r for r in role_records if r["role"] == role]
        agg["n_" + role] = len(rs)
        agg[role + "_vmax"] = round(max((r["vmax"] for r in rs), default=np.nan), 1)
    att = [r for r in role_records if r["role"] == "attacker"]
    agg["attacker_approach_cms"] = round(
        max((r["approach_speed"] for r in att), default=np.nan), 1)
    lib = [r for r in role_records if r["role"] == "libero"]
    agg["libero_lateral_cms"] = round(
        max((r["lateral_speed"] for r in lib), default=np.nan), 1)
    st = [r for r in role_records if r["role"] == "setter"]
    agg["setter_displacement_cm"] = round(
        min((r["displacement"] for r in st), default=np.nan), 1)
    agg["n_unknown"] = sum(1 for r in role_records if r["role"] == UNKNOWN)
    return agg


__all__ = ["ROLES", "UNKNOWN", "ROLE_LITERATURE", "slot_kinematics",
           "infer_roles", "role_skill_summary", "NET_Y",
           "ATTACK_ZONE_CM", "NEAR_NET_CM", "BACK_ROW_CM"]
