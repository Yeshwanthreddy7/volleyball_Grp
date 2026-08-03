"""Tests for class-id resolution - the custom-model loophole guard. NumPy-free."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detect_utils import resolve_class_ids  # noqa: E402


def test_coco_default():
    names = {0: "person", 32: "sports ball"}
    assert resolve_class_ids(names) == (0, 32)


def test_roboflow_alphabetical_2class():
    # Roboflow orders alphabetically: ball=0, player=1  (the dangerous case)
    names = {0: "ball", 1: "player"}
    assert resolve_class_ids(names) == (1, 0)


def test_volleyball_naming():
    names = {0: "Volleyball", 1: "Player"}
    assert resolve_class_ids(names) == (1, 0)


def test_single_class_player_only():
    names = {0: "player"}
    assert resolve_class_ids(names) == (0, None)


def test_overrides_win():
    names = {0: "ball", 1: "player"}
    assert resolve_class_ids(names, person_override=1, ball_override=0) == (1, 0)
    assert resolve_class_ids(names, person_override=5)[0] == 5


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print("  PASS  " + fn.__name__)
    print("\n{}/{} tests passed.".format(len(fns), len(fns)))


if __name__ == "__main__":
    _run_all()
