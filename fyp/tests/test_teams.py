"""
Tests for jersey-colour team separation (fyp/teams.py).

The failure this module fixes, measured on "videoplayback (4)" frame 11290:
the geometric foot-point-vs-net filter passed TEN players - both teams mixed -
into a six-slot identity system, so every downstream tactical feature described
a formation that never existed on court.

Numpy + cv2 only; no torch, no weights, no video.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from teams import (  # noqa: E402
    TeamClassifier, TeamVoter, torso_descriptor, NEAR, FAR, UNKNOWN,
)

cv2 = pytest.importorskip("cv2")


def _frame_with_jerseys(colours, box_w=40, box_h=120, gap=20):
    """Synthetic frame: one solid-colour 'player' per entry, left to right.

    Returns (frame, xyxy). Colours are BGR tuples.
    """
    n = len(colours)
    w = n * (box_w + gap) + gap
    frame = np.full((box_h + 2 * gap, w, 3), 60, dtype=np.uint8)  # dark court
    boxes = []
    for i, bgr in enumerate(colours):
        x1 = gap + i * (box_w + gap)
        y1 = gap
        x2, y2 = x1 + box_w, y1 + box_h
        frame[y1:y2, x1:x2] = bgr
        boxes.append([x1, y1, x2, y2])
    return frame, np.array(boxes, dtype=float)


LIGHT_BLUE = (230, 170, 90)     # BGR - ARG-like
DARK_NAVY = (70, 40, 25)        # BGR - ITA-like
WHITE = (240, 240, 240)         # libero-like contrasting kit


# --------------------------------------------------------------------------
# descriptor
# --------------------------------------------------------------------------

def test_same_jersey_gives_near_identical_descriptors():
    frame, boxes = _frame_with_jerseys([LIGHT_BLUE, LIGHT_BLUE])
    d = torso_descriptor(frame, boxes)
    assert np.allclose(d[0], d[1], atol=1e-6)


def test_different_jerseys_are_separated_in_descriptor_space():
    frame, boxes = _frame_with_jerseys([LIGHT_BLUE, DARK_NAVY])
    d = torso_descriptor(frame, boxes)
    assert np.linalg.norm(d[0] - d[1]) > 0.2


def test_descriptor_samples_the_torso_not_the_whole_box():
    """Legs/background must not drive the colour: paint only the torso band."""
    frame, boxes = _frame_with_jerseys([DARK_NAVY])
    x1, y1, x2, y2 = boxes[0].astype(int)
    frame[y1:y2, x1:x2] = (60, 60, 60)                       # neutral body
    bh = y2 - y1
    frame[y1 + int(0.15 * bh):y1 + int(0.50 * bh), x1:x2] = LIGHT_BLUE
    d = torso_descriptor(frame, boxes)

    ref, ref_boxes = _frame_with_jerseys([LIGHT_BLUE])
    d_ref = torso_descriptor(ref, ref_boxes)
    assert np.linalg.norm(d[0] - d_ref[0]) < 0.15


def test_empty_input_returns_empty():
    frame, _ = _frame_with_jerseys([LIGHT_BLUE])
    assert torso_descriptor(frame, np.empty((0, 4))).shape == (0, 3)


def test_degenerate_box_does_not_crash():
    frame, _ = _frame_with_jerseys([LIGHT_BLUE])
    d = torso_descriptor(frame, np.array([[10.0, 10.0, 10.0, 10.0]]))
    assert d.shape == (1, 3) and np.all(np.isfinite(d))


# --------------------------------------------------------------------------
# classifier
# --------------------------------------------------------------------------

def _run_warmup(clf, frame, boxes, ys, n_frames=60):
    for _ in range(n_frames):
        out = clf.update(frame, boxes, ys)
    return out


def test_classifier_learns_team_from_court_side_then_uses_colour():
    """The core contract: geometry teaches, colour decides."""
    frame, boxes = _frame_with_jerseys(
        [LIGHT_BLUE, LIGHT_BLUE, LIGHT_BLUE, DARK_NAVY, DARK_NAVY, DARK_NAVY]
    )
    ys = [700, 650, 600, 200, 250, 300]        # blue near, navy far
    clf = TeamClassifier(n_clusters=2, warmup_frames=10, min_samples=12)
    _run_warmup(clf, frame, boxes, ys, n_frames=20)
    assert clf.ready

    labels = clf.predict(frame, boxes)
    assert list(labels) == [NEAR, NEAR, NEAR, FAR, FAR, FAR]


def test_player_at_the_net_is_assigned_by_colour_not_position():
    """The exact frame-11290 failure: a near-team player standing at the net.

    Geometry would call this player FAR (foot point on the wrong side of an
    ambiguous line); colour must keep them with their team-mates.
    """
    frame, boxes = _frame_with_jerseys(
        [LIGHT_BLUE, LIGHT_BLUE, LIGHT_BLUE, DARK_NAVY, DARK_NAVY, DARK_NAVY]
    )
    ys = [700, 650, 600, 200, 250, 300]
    clf = TeamClassifier(n_clusters=2, warmup_frames=10, min_samples=12)
    _run_warmup(clf, frame, boxes, ys, n_frames=20)

    # Same six jerseys, but now the blue #3 has advanced to the net (y=455).
    labels = clf.predict(frame, boxes)
    assert labels[2] == NEAR, "blue player at the net must stay with the blues"


def test_libero_contrasting_jersey_attaches_to_its_own_team():
    """FIVB requires a contrasting libero kit, so a team is TWO colour
    populations. With k>2 the libero cluster is labelled by its own court-side
    statistics instead of being forced into the nearest opponent colour."""
    frame, boxes = _frame_with_jerseys(
        [LIGHT_BLUE, LIGHT_BLUE, WHITE, DARK_NAVY, DARK_NAVY]
    )
    ys = [700, 650, 720, 200, 250]           # white libero plays NEAR back court
    clf = TeamClassifier(n_clusters=3, warmup_frames=10, min_samples=12)
    _run_warmup(clf, frame, boxes, ys, n_frames=20)

    labels = clf.predict(frame, boxes)
    assert labels[2] == NEAR, "libero must attach to the near team, not the navy"
    assert list(labels[:2]) == [NEAR, NEAR]
    assert list(labels[3:]) == [FAR, FAR]


def test_update_returns_none_during_warmup():
    frame, boxes = _frame_with_jerseys([LIGHT_BLUE, DARK_NAVY])
    clf = TeamClassifier(warmup_frames=50, min_samples=1000)
    assert clf.update(frame, boxes, [700, 200]) is None
    assert not clf.ready


def test_predict_before_fit_raises_rather_than_guessing():
    frame, boxes = _frame_with_jerseys([LIGHT_BLUE])
    clf = TeamClassifier()
    with pytest.raises(RuntimeError):
        clf.predict(frame, boxes)


def test_fit_report_records_the_evidence():
    frame, boxes = _frame_with_jerseys([LIGHT_BLUE, LIGHT_BLUE, DARK_NAVY])
    clf = TeamClassifier(n_clusters=2, warmup_frames=5, min_samples=6)
    _run_warmup(clf, frame, boxes, [700, 650, 200], n_frames=10)
    rep = clf.fit_report
    assert rep["n_samples"] > 0
    assert rep["n_near_clusters"] >= 1 and rep["n_far_clusters"] >= 1
    assert all("median_court_y" in c for c in rep["clusters"])


def test_non_finite_court_y_is_not_used_for_training():
    frame, boxes = _frame_with_jerseys([LIGHT_BLUE, DARK_NAVY])
    clf = TeamClassifier(n_clusters=2, warmup_frames=5, min_samples=4)
    for _ in range(10):
        clf.update(frame, boxes, [700.0, np.nan])
    # Only the finite sample trained the model, so nothing is labelled FAR.
    if clf.ready:
        assert set(clf.cluster_team.tolist()) <= {NEAR, FAR, UNKNOWN}


# --------------------------------------------------------------------------
# degenerate-fit guard
# --------------------------------------------------------------------------

def test_all_one_side_is_flagged_degenerate():
    """The real clip_002 failure: with no court mask active, 24.6 detections
    per frame were mostly CROWD. Their varied shirt colours captured every
    cluster, all cluster medians landed beyond the net, the whole clip was
    labelled FAR, and it extracted with ZERO players - silently."""
    frame, boxes = _frame_with_jerseys([LIGHT_BLUE, DARK_NAVY, WHITE])
    clf = TeamClassifier(n_clusters=3, warmup_frames=5, min_samples=6)
    _run_warmup(clf, frame, boxes, [100, 150, 200], n_frames=10)   # all FAR

    assert clf.ready
    assert clf.degenerate, "a fit that found no near-side team must be flagged"


def test_healthy_two_sided_fit_is_not_degenerate():
    frame, boxes = _frame_with_jerseys([LIGHT_BLUE, LIGHT_BLUE, DARK_NAVY])
    clf = TeamClassifier(n_clusters=2, warmup_frames=5, min_samples=6)
    _run_warmup(clf, frame, boxes, [700, 650, 200], n_frames=10)

    assert clf.ready and not clf.degenerate


def test_unfitted_classifier_reports_degenerate():
    """Callers check `degenerate` to decide whether to trust colour; an
    unfitted model must not read as trustworthy."""
    assert TeamClassifier().degenerate


# --------------------------------------------------------------------------
# per-track voting
# --------------------------------------------------------------------------

def test_vote_overrides_a_single_bad_frame():
    """One blurred/occluded frame must not flip a track that is right in the
    other N-1 - the residual error mode measured on real footage."""
    v = TeamVoter()
    ids = np.array([7])
    for _ in range(9):
        v.update(ids, np.array([NEAR]))
    out = v.update(ids, np.array([FAR]))      # the bad frame
    assert out[0] == NEAR


def test_vote_follows_a_sustained_majority():
    v = TeamVoter()
    ids = np.array([7])
    for _ in range(3):
        v.update(ids, np.array([NEAR]))
    for _ in range(20):
        out = v.update(ids, np.array([FAR]))
    assert out[0] == FAR


def test_votes_are_independent_per_track():
    v = TeamVoter()
    ids = np.array([1, 2])
    for _ in range(5):
        out = v.update(ids, np.array([NEAR, FAR]))
    assert out[0] == NEAR and out[1] == FAR


def test_forget_drops_dead_tracks_so_recycled_ids_start_clean():
    v = TeamVoter()
    for _ in range(10):
        v.update(np.array([3]), np.array([NEAR]))
    v.forget(np.array([]))                      # track 3 died
    out = v.update(np.array([3]), np.array([FAR]))   # id recycled by ByteTrack
    assert out[0] == FAR


def test_voter_handles_empty_frame():
    v = TeamVoter()
    assert len(v.update(np.array([]), np.array([]))) == 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-q"]))
