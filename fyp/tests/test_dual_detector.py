"""
Tests for DualDetector - the split-backend fix (technical review §13).

Root cause it guards against: the 416-image Roboflow fine-tune detects the BALL
well but collapses on PLAYERS in unseen footage (measured 0.1 players/frame on
"videoplayback (4)", vs 23.3 for stock yolo11n on the identical frames). The
pipeline must therefore be able to take players from one backend and the ball
from another, and must never silently prefer the weaker source.

No ultralytics/torch needed: DualDetector composes BaseDetector objects, so the
composition logic is tested with fakes.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from interfaces import DetectionResult, BaseDetector, DualDetector  # noqa: E402


class FakeDetector(BaseDetector):
    """Minimal BaseDetector returning a scripted result."""

    def __init__(self, n_players=0, ball=None, tag="fake", class_names=None,
                 person_id=0, ball_id=None):
        self._n = n_players
        self._ball = None if ball is None else np.asarray(ball, dtype=float)
        self._tag = tag
        self.class_names = class_names or {0: "person"}
        self.person_id = person_id
        self.ball_id = ball_id
        self.calls = 0

    @property
    def name(self):
        return self._tag

    def predict(self, frame):
        self.calls += 1
        return DetectionResult(
            person_xyxy=np.tile([10.0, 20.0, 30.0, 60.0], (self._n, 1)),
            person_conf=np.full((self._n,), 0.9),
            ball_center=None if self._ball is None else self._ball.copy(),
        )


FRAME = np.zeros((720, 1280, 3), dtype=np.uint8)


def test_players_come_from_player_backend_not_ball_backend():
    """The real failure mode: the ball model reports ~0 players; its player
    boxes must never reach the tracker."""
    players = FakeDetector(n_players=20, tag="stock")
    ball = FakeDetector(n_players=0, ball=(781.0, 193.0), tag="custom", ball_id=0)
    det = DualDetector(players, ball)

    out = det.predict(FRAME)
    assert len(out.person_xyxy) == 20
    assert len(out.person_conf) == 20


def test_ball_comes_from_ball_backend():
    players = FakeDetector(n_players=20, ball=None, tag="stock")
    ball = FakeDetector(n_players=0, ball=(781.0, 193.0), tag="custom", ball_id=0)
    det = DualDetector(players, ball)

    out = det.predict(FRAME)
    assert out.ball_center is not None
    assert np.allclose(out.ball_center, [781.0, 193.0])


def test_ball_backend_wins_over_player_backend_ball():
    """Both backends see a ball -> the dedicated model is the authority."""
    players = FakeDetector(n_players=12, ball=(100.0, 100.0), tag="stock")
    ball = FakeDetector(n_players=0, ball=(781.0, 193.0), tag="custom", ball_id=0)
    det = DualDetector(players, ball)

    assert np.allclose(det.predict(FRAME).ball_center, [781.0, 193.0])


def test_falls_back_to_player_backend_ball_when_ball_model_misses():
    """Ball recall is ~67% at best, so the fallback path is the common path."""
    players = FakeDetector(n_players=12, ball=(100.0, 100.0), tag="stock")
    ball = FakeDetector(n_players=0, ball=None, tag="custom", ball_id=0)
    det = DualDetector(players, ball)

    assert np.allclose(det.predict(FRAME).ball_center, [100.0, 100.0])


def test_no_ball_anywhere_is_none_not_a_fabricated_point():
    players = FakeDetector(n_players=12, ball=None, tag="stock")
    ball = FakeDetector(n_players=0, ball=None, tag="custom", ball_id=0)
    assert DualDetector(players, ball).predict(FRAME).ball_center is None


def test_single_backend_mode_is_passthrough():
    players = FakeDetector(n_players=7, ball=(5.0, 6.0), tag="stock", ball_id=32)
    det = DualDetector(players, None)

    out = det.predict(FRAME)
    assert len(out.person_xyxy) == 7
    assert np.allclose(out.ball_center, [5.0, 6.0])
    assert det.ball_id == 32          # capability advertised from the one backend


def test_ball_capability_advertised_from_ball_backend():
    """pipeline.py disables the ball overlay when ball_id is None - so the dual
    detector must report the ball backend's id, not the player backend's."""
    players = FakeDetector(n_players=12, tag="stock", ball_id=None)
    ball = FakeDetector(n_players=0, ball=(1.0, 2.0), tag="custom", ball_id=0)
    assert DualDetector(players, ball).ball_id == 0


def test_both_backends_are_invoked_once_per_frame():
    players = FakeDetector(n_players=12, tag="stock")
    ball = FakeDetector(n_players=0, ball=(1.0, 2.0), tag="custom", ball_id=0)
    det = DualDetector(players, ball)

    det.predict(FRAME)
    det.predict(FRAME)
    assert players.calls == 2 and ball.calls == 2


def test_name_reports_both_backends_for_the_run_log():
    det = DualDetector(FakeDetector(tag="stock@1280"),
                       FakeDetector(tag="custom@1280", ball_id=0))
    assert "stock@1280" in det.name and "custom@1280" in det.name


def test_default_imgsz_is_high_resolution():
    """Regression lock on the 640->1280 fix: ball recall 15% -> 77%."""
    from interfaces import DEFAULT_IMGSZ
    assert DEFAULT_IMGSZ >= 1280


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-q"]))
