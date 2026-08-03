"""Tests for the online SlotManager occlusion bridge. Numpy-only."""
import os
import sys

import numpy as np  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from identity import SlotManager  # noqa: E402


def test_stable_ids_keep_slots():
    sm = SlotManager()
    a = sm.assign({1: (100, 500), 2: (300, 500)})
    for _ in range(20):
        b = sm.assign({1: (100, 500), 2: (300, 500)})
    assert a[1] == b[1] and a[2] == b[2] and a[1] != a[2]


def test_bridge_same_slot_after_short_occlusion_new_id():
    """Player occluded 5 frames, tracker issues NEW id near old position ->
    inherits the SAME slot (the actual 'id not consistent' failure mode)."""
    sm = SlotManager()
    m = sm.assign({7: (400, 600)})
    slot7 = m[7]
    for _ in range(5):
        sm.assign({})                       # occlusion: nothing visible
    m = sm.assign({99: (430, 610)})         # new id, ~32 cm away
    assert m[99] == slot7
    assert sm.n_bridges == 1


def test_far_reappearance_gets_new_slot():
    sm = SlotManager()
    m = sm.assign({7: (400, 600)})
    slot7 = m[7]
    sm.assign({})
    m = sm.assign({99: (1400, 600)})        # 1000 cm away: not the same player
    assert m[99] != slot7


def test_gate_grows_with_gap_but_is_capped():
    sm = SlotManager(gate_cm_per_frame=60, gate_cap_cm=250)
    sm.assign({1: (400, 600)})
    for _ in range(10):
        sm.assign({})                       # 10-frame gap -> gate = 250 (cap)
    m = sm.assign({50: (640, 600)})         # 240 cm moved: within cap
    assert m[50] == 0
    sm2 = SlotManager(gate_cm_per_frame=60, gate_cap_cm=250)
    sm2.assign({1: (400, 600)})
    sm2.assign({})                          # 1 hidden frame = 2 motion intervals
    m2 = sm2.assign({50: (540, 600)})       # 140 cm > gate 120: too fast
    assert m2[50] != 0


def test_slot_expires_after_max_gap():
    sm = SlotManager(max_gap=15)
    sm.assign({7: (400, 600)})
    for _ in range(16):
        sm.assign({})
    m = sm.assign({99: (405, 600)})         # gap 16 > 15: no bridge, but slot is free
    assert m[99] == 0 and sm.n_bridges == 0


def test_limbo_slot_not_stolen_by_far_new_player():
    """While a slot is in limbo its column must not be recycled by a distant
    new player - that is exactly the column-churn bug being fixed."""
    sm = SlotManager(n_slots=2)
    m = sm.assign({1: (100, 500), 2: (900, 500)})
    s1 = m[1]
    sm.assign({2: (900, 500)})              # player 1 occluded (limbo)
    m = sm.assign({2: (900, 500), 33: (1700, 800)})  # far newcomer
    assert 33 not in m                      # both slots reserved -> ignored
    m = sm.assign({2: (900, 500), 44: (120, 505)})   # near old pos -> bridge
    assert m[44] == s1


def test_two_new_ids_nearest_wins():
    sm = SlotManager()
    sm.assign({1: (100, 500), 2: (800, 500)})
    sm.assign({})                           # both in limbo
    m = sm.assign({10: (110, 500), 11: (790, 505)})
    assert m[10] == 0 and m[11] == 1        # each bridged to its own slot


def test_seventh_player_ignored():
    sm = SlotManager()
    feet = {i: (100.0 + 250 * i, 600.0) for i in range(1, 7)}
    m = sm.assign(feet)
    assert len(set(m.values())) == 6
    feet[7] = (1750.0, 850.0)
    m = sm.assign(feet)
    assert 7 not in m                       # 6-player cap holds


def test_nan_positions_ignored():
    sm = SlotManager()
    m = sm.assign({1: (float("nan"), 500), 2: (300, 500)})
    assert 1 not in m and 2 in m


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print("  PASS  " + fn.__name__)
    print("\n{}/{} tests passed.".format(len(fns), len(fns)))


if __name__ == "__main__":
    _run_all()
