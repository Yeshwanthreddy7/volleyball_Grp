"""Tests for the class-weight collapse guard. Numpy-only."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_utils import class_weights_from_counts  # noqa: E402


def test_empty_class_gets_zero_weight_exact_kaggle_case():
    # The exact split that collapsed: Spacing 42, Delayed 23, Attack 82,
    # Defense 101, Unclassified 0.
    w = class_weights_from_counts([42, 23, 82, 101, 0])
    assert w[4] == 0.0                       # empty class: NO pull
    assert w[1] == w.max()                   # rarest present class boosted most
    present = w[:4]
    assert abs(present.mean() - 1.0) < 1e-9  # LR-preserving normalisation
    # the old formula gave the empty class ~4.8 - two orders too high
    old = 5.0 / (np.array([42, 23, 82, 101, 0]) + 1.0)
    old = old / old.mean()
    assert old[4] > 4.0                      # documents the bug being guarded


def test_balanced_counts_give_unit_weights():
    w = class_weights_from_counts([50, 50, 50, 50])
    assert np.allclose(w, 1.0)


def test_cap_limits_minority_boost():
    # normalised inverse-frequency weight is bounded by K (num classes), so
    # the cap only binds for K >= 5 - exactly our 5-class problem.
    w = class_weights_from_counts([1, 1000, 1000, 1000, 1000], cap=4.0)
    assert w.max() <= 4.0 and w[0] == 4.0


def test_all_empty_is_safe():
    assert np.allclose(class_weights_from_counts([0, 0, 0]), 0.0)


def test_video_key_parsing():
    from train_utils import video_key
    assert video_key("Coordinated_Attack__videoplayback (1)_clip_002.csv") == "(1)"
    assert video_key("Coordinated_Defense__videoplayback (3)_clip_215.csv") == "(3)"
    assert video_key("Delayed_Support__videoplayback_clip_306.csv") == "(plain)"
    assert video_key("something_else.csv") == "?"


def test_holdout_indices_partition():
    from train_utils import holdout_indices
    files = ["A__videoplayback (1)_clip_1.csv", "B__videoplayback (3)_clip_2.csv",
             "C__videoplayback_clip_3.csv", "D__videoplayback (1)_clip_4.csv"]
    test, rest = holdout_indices(files, "(1)")
    assert test == [0, 3] and rest == [1, 2]
    test2, rest2 = holdout_indices(files, "plain")     # bare key also accepted
    assert test2 == [2] and sorted(test2 + rest2) == [0, 1, 2, 3]


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print("  PASS  " + fn.__name__)
    print("\n{}/{} tests passed.".format(len(fns), len(fns)))


if __name__ == "__main__":
    _run_all()
