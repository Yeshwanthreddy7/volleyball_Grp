"""Regression test: the Kalman Q-scale patch must be IDEMPOTENT.

The bug it guards against: shared_kalman is a class-level global, trackers are
rebuilt per clip, and the old patch re-wrapped the CURRENT class each time -
stacking N wrapper layers over N clips, compounding Q by 4^N (overflow
warnings) and finally raising RecursionError mid-extraction on Kaggle.
Torch/ultralytics-free: uses fake KF/track classes."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tracking import _patch_kalman  # noqa: E402


def _fresh_fakes():
    class FakeKF:
        _std_weight_position = 1.0 / 20
        _std_weight_velocity = 1.0 / 160

        def predict(self, mean, covariance):
            return self._std_weight_position, self._std_weight_velocity

        def multi_predict(self, mean, covariance):
            return self._std_weight_position, self._std_weight_velocity

    class FakeSTrack:
        shared_kalman = FakeKF()

    class FakeTracker:
        def __init__(self):
            self.kalman_filter = FakeSTrack.shared_kalman

    return FakeKF, FakeSTrack, FakeTracker


def test_single_patch_scales_q_during_call_only():
    FakeKF, FakeSTrack, FakeTracker = _fresh_fakes()
    assert _patch_kalman(FakeSTrack, FakeTracker(), 4.0, "test")
    kf = FakeSTrack.shared_kalman
    wp, wv = kf.predict(None, None)          # inside the call: scaled by sqrt(4)=2
    assert abs(wp - 2.0 / 20) < 1e-12 and abs(wv - 2.0 / 160) < 1e-12
    assert abs(kf._std_weight_position - 1.0 / 20) < 1e-12   # restored after


def test_repatching_300_times_does_not_stack_layers():
    FakeKF, FakeSTrack, FakeTracker = _fresh_fakes()
    _patch_kalman(FakeSTrack, FakeTracker(), 4.0, "test")
    tuned_cls = type(FakeSTrack.shared_kalman)
    for _ in range(300):                     # 300 clips worth of tracker builds
        _patch_kalman(FakeSTrack, FakeTracker(), 4.0, "test")
    assert type(FakeSTrack.shared_kalman) is tuned_cls       # same class, no wrap
    wp, _ = FakeSTrack.shared_kalman.multi_predict(None, None)
    assert abs(wp - 2.0 / 20) < 1e-12        # exactly x2, NOT 2**300 / overflow


def test_different_scale_rebuilds_from_pristine_base():
    FakeKF, FakeSTrack, FakeTracker = _fresh_fakes()
    _patch_kalman(FakeSTrack, FakeTracker(), 4.0, "test")
    _patch_kalman(FakeSTrack, FakeTracker(), 9.0, "test")
    cls = type(FakeSTrack.shared_kalman)
    assert cls._fyp_base_cls is FakeKF       # base, not the q4 wrapper
    wp, _ = FakeSTrack.shared_kalman.predict(None, None)
    assert abs(wp - 3.0 / 20) < 1e-12        # sqrt(9)=3, applied once


def test_tracker_instance_mirrors_tuned_filter():
    FakeKF, FakeSTrack, FakeTracker = _fresh_fakes()
    t1, t2 = FakeTracker(), FakeTracker()
    _patch_kalman(FakeSTrack, t1, 4.0, "test")
    _patch_kalman(FakeSTrack, t2, 4.0, "test")   # idempotent path
    assert type(t2.kalman_filter) is type(FakeSTrack.shared_kalman)


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print("  PASS  " + fn.__name__)
    print("\n{}/{} tests passed.".format(len(fns), len(fns)))


if __name__ == "__main__":
    _run_all()
