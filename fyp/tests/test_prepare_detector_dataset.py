"""Tests for the leak-free Roboflow dataset splitter. Stdlib only."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prepare_detector_dataset import (  # noqa: E402
    audit_label_file, ensure_class_presence, parse_frame_number,
    plan_gap_exclusions, plan_split, read_names_from_yaml, render_data_yaml,
)


def test_parse_frame_number():
    assert parse_frame_number("frame_0123_jpg.rf.abc") == 123
    assert parse_frame_number("frame-7") == 7
    assert parse_frame_number("clip99_x") == 99
    assert parse_frame_number("noneheretoparse") is None


def test_plan_split_contiguous_and_deterministic():
    stems = ["frame_{:04d}".format(i * 3) for i in range(100)]
    chunks, val_idx = plan_split(stems, val_fraction=0.15, n_blocks=12)
    assert sum(len(c) for c in chunks) == 100
    flat = [s for c in chunks for s in c]
    assert flat == sorted(stems, key=parse_frame_number)   # temporal order kept
    assert len(val_idx) == 2                               # round(0.15*12)
    assert all(0 <= i < 12 for i in val_idx)
    chunks2, val_idx2 = plan_split(list(reversed(stems)), 0.15, 12)
    assert val_idx == val_idx2 and chunks == chunks2       # deterministic


def test_val_blocks_are_spread_apart():
    stems = ["frame_{:04d}".format(i) for i in range(120)]
    _, val_idx = plan_split(stems, val_fraction=0.15, n_blocks=12)
    assert len(val_idx) >= 2 and (max(val_idx) - min(val_idx)) >= 3


def test_ensure_class_presence_swaps_when_ball_missing():
    chunks = [["a"], ["b"], ["c"], ["d"]]
    counts = {"a": {1: 5}, "b": {1: 5}, "c": {0: 3, 1: 2}, "d": {1: 4}}
    fixed = ensure_class_presence(chunks, [0], counts, cls=0)
    assert 2 in fixed and 0 not in fixed                   # swapped in block 'c'
    same = ensure_class_presence(chunks, [2], counts, cls=0)
    assert same == [2]                                     # already present


def test_gap_exclusions_quarantine_neighbours_only():
    ordered = ["f{}".format(i) for i in range(10)]
    val = {"f4", "f5"}
    ex = plan_gap_exclusions(ordered, val, gap=1)
    assert ex == {"f3", "f6"}
    assert not (ex & val)


def test_audit_label_file_polygon_box_bad():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "x.txt")
        with open(p, "w") as fh:
            fh.write("1 0.5 0.5 0.1 0.2\n")                       # box
            fh.write("0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n")       # polygon
            fh.write("1 0.5 0.5 0.1\n")                           # bad arity
            fh.write("1 0.5 1.7 0.1 0.2\n")                       # out of range
        a = audit_label_file(p)
        assert a["box"] == 1 and a["poly"] == 1 and a["bad"] == 2
        assert a["counts"] == {1: 1, 0: 1}


def test_yaml_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        y = os.path.join(td, "data.yaml")
        with open(y, "w") as fh:
            fh.write("train: ../train/images\nnc: 2\nnames: ['ball', 'player']\n")
        names = read_names_from_yaml(y)
        assert names == ["ball", "player"]
        text = render_data_yaml(td, names)
        assert "path: " in text and "val: valid/images" in text
        assert "0: ball" in text and "1: player" in text and "nc: 2" in text


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print("  PASS  " + fn.__name__)
    print("\n{}/{} tests passed.".format(len(fns), len(fns)))


if __name__ == "__main__":
    _run_all()
