"""Tests for behavioural role inference. Numpy-only, deterministic."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from roles import UNKNOWN, infer_roles, role_skill_summary, slot_kinematics  # noqa: E402

T, FPS = 29, 30.0


def _traj(points):
    """Linear interpolation through waypoints [(t, x, y), ...] -> (T,2)."""
    p = np.full((T, 2), np.nan)
    pts = sorted(points)
    for (t0, x0, y0), (t1, x1, y1) in zip(pts[:-1], pts[1:]):
        for t in range(t0, t1 + 1):
            a = (t - t0) / max(t1 - t0, 1)
            p[t] = [x0 + a * (x1 - x0), y0 + a * (y1 - y0)]
    return p


def _team():
    """Six synthetic bottom-side players with textbook role signatures."""
    slots = np.full((T, 6, 2), np.nan)
    # 0: attacker - explosive approach 850->470 (back to net) in ~0.6 s
    slots[:, 0, :] = _traj([(0, 600, 850), (10, 600, 840), (28, 600, 470)])
    # 1: setter - parked at the net, tiny jitter
    slots[:, 1, :] = _traj([(0, 950, 500), (28, 960, 505)])
    # 2: libero - deep, fast lateral sweep (~600 cm/s, below the 800 cap)
    slots[:, 2, :] = _traj([(0, 500, 860), (28, 1060, 855)])
    # 3: defender - deep, mild drift
    slots[:, 3, :] = _traj([(0, 1500, 800), (28, 1460, 780)])
    # 4: second attacker - approach but stops outside attack zone
    slots[:, 4, :] = _traj([(0, 1200, 880), (28, 1200, 650)])
    # 5: mostly occluded (low visibility -> unknown)
    slots[5:9, 5, :] = [[300, 700]] * 4
    return slots


def test_textbook_roles_recovered():
    rec = infer_roles(_team(), fps=FPS, team_side="bottom")
    assert rec[0]["role"] == "attacker"
    assert rec[1]["role"] == "setter"
    assert rec[2]["role"] == "libero"
    assert rec[3]["role"] == "defender"
    assert rec[5]["role"] == UNKNOWN          # 4/29 frames visible


def test_attacker_approach_speed_in_elite_band():
    rec = infer_roles(_team(), fps=FPS, team_side="bottom")
    v = rec[0]["approach_speed"]
    # 370 cm over ~0.6 s ~ 600 cm/s peak segment; must exceed elite floor
    assert v > 260.0, v


def test_exclusivity_one_setter_one_libero():
    slots = _team()
    # clone the setter into slot 3 - two net-parked players
    slots[:, 3, :] = slots[:, 1, :] + np.array([80.0, 0.0])
    rec = infer_roles(slots, fps=FPS, team_side="bottom")
    assert sum(1 for r in rec if r["role"] == "setter") <= 1
    assert sum(1 for r in rec if r["role"] == "libero") <= 1


def test_top_side_mirroring():
    slots = _team()
    mirrored = slots.copy()
    mirrored[:, :, 1] = 900.0 - mirrored[:, :, 1]   # reflect to top half
    rec = infer_roles(mirrored, fps=FPS, team_side="top")
    assert rec[0]["role"] == "attacker" and rec[2]["role"] == "libero"


def test_glitch_speeds_capped():
    slots = _team()
    slots[14, 0, :] = [600, 100]                    # teleport (tracking glitch)
    slots[15, 0, :] = [600, 660]
    kin = slot_kinematics(slots, fps=FPS, team_side="bottom")
    assert kin[0]["vmax"] <= 800.0                  # physio cap respected


def test_all_nan_slot_is_unknown_and_summary_counts():
    slots = _team()
    slots[:, 5, :] = np.nan
    rec = infer_roles(slots, fps=FPS, team_side="bottom")
    assert rec[5]["role"] == UNKNOWN
    agg = role_skill_summary(rec)
    assert agg["n_attacker"] >= 1 and agg["n_unknown"] >= 1
    assert agg["attacker_approach_cms"] > 260.0


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print("  PASS  " + fn.__name__)
    print("\n{}/{} tests passed.".format(len(fns), len(fns)))


if __name__ == "__main__":
    _run_all()
