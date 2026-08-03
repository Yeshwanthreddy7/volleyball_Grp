"""Tests for the dual-interface detection shim (no ultralytics needed)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tracking import _BoxesShim  # noqa: E402


def test_attributes_and_len():
    s = _BoxesShim.from_xyxy([[0, 0, 10, 20], [5, 5, 15, 25]], [0.9, 0.6])
    assert len(s) == 2
    assert s.xywh.shape == (2, 4) and s.xyxy.shape == (2, 4)
    assert np.allclose(s.xywh[0], [5, 10, 10, 20])
    assert np.allclose(s.conf, [0.9, 0.6]) and np.allclose(s.cls, 0)


def test_boolean_mask_subscript_like_new_bytetracker():
    s = _BoxesShim.from_xyxy([[0, 0, 10, 20], [5, 5, 15, 25], [1, 1, 9, 9]],
                             [0.9, 0.3, 0.05])
    hi = s.conf >= 0.5
    lo = (s.conf > 0.1) & (~hi)
    high, low = s[hi], s[lo]           # exactly what _split_detections does
    assert len(high) == 1 and np.allclose(high.conf, [0.9])
    assert len(low) == 1 and np.allclose(low.conf, [0.3])
    assert high.xywh.shape == (1, 4)


def test_int_array_and_chained_subscript():
    s = _BoxesShim.from_xyxy([[0, 0, 2, 2], [1, 1, 3, 3], [2, 2, 4, 4]],
                             [0.5, 0.6, 0.7])
    sub = s[np.array([2, 0])]
    assert np.allclose(sub.conf, [0.7, 0.5])
    sub2 = sub[np.array([True, False])]
    assert len(sub2) == 1 and np.allclose(sub2.conf, [0.7])


def test_empty_shim():
    e = _BoxesShim.empty()
    assert len(e) == 0 and e.xywh.shape == (0, 4)
    assert len(e[np.array([], dtype=bool)]) == 0


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print("  PASS  " + fn.__name__)
    print("\n{}/{} tests passed.".format(len(fns), len(fns)))


if __name__ == "__main__":
    _run_all()
