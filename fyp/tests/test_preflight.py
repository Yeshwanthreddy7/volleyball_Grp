"""
Tests for the preflight gate (fyp/preflight.py).

Each of the first three tests replays an ACTUAL historical failure of this
project, using the populations that were really measured at the time. Every one
of them ran to completion and printed a formatted tactical report. The gate's
job is to make each of them stop instead.

Pure arithmetic on a stats dict - no video, no weights, no cv2, no torch.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preflight import (  # noqa: E402
    assess, summarise, FATAL, WARN,
    MIN_TEAM_MEDIAN, MAX_TEAM_MEDIAN,
)


def _stats(raw=20.0, masked=10.0, team=6.0, ball=0.7, n=24):
    return {
        "n_frames": n,
        "raw_person_median": raw,
        "masked_person_median": masked,
        "team_person_median": team,
        "ball_recall": ball,
    }


# --------------------------------------------------------------------------
# the three real regressions
# --------------------------------------------------------------------------

def test_regression_13_detector_blind_to_players():
    """§13: the volleyball fine-tune scored 0.06 max on a frame with twelve
    visible players -> 0.1 detections/frame, zero boxes drawn, 100% TRANSITION,
    and a cheerful 'Team Coordination Analysis' at the end."""
    v = assess(_stats(raw=0.0, masked=0.0, team=0.0, ball=0.05))
    assert v.fatal
    assert v.findings[0].stage == "detection"
    assert "yolo11n" in v.findings[0].remedy


def test_regression_128_court_mask_deletes_everyone():
    """§12.8: a half-court quad mapped onto the full-court plane put the net at
    y=0, so the team-side filter deleted the entire front row."""
    v = assess(_stats(raw=21.0, masked=0.0, team=0.0))
    assert v.fatal
    assert v.findings[0].stage == "court mask"


def test_regression_129_team_split_starves_the_slots():
    """§12.9: off-court coaches held the identity slots; the real players never
    reached the analysis."""
    v = assess(_stats(raw=20.0, masked=8.0, team=0.0))
    assert v.fatal
    assert v.findings[0].stage == "team split"


# --------------------------------------------------------------------------
# the healthy case
# --------------------------------------------------------------------------

def test_healthy_run_passes_cleanly():
    v = assess(_stats(raw=20.0, masked=10.0, team=6.0, ball=0.7))
    assert not v.fatal
    assert v.findings == []
    assert "All preflight checks passed" in v.render()


def test_seven_players_is_acceptable_not_an_error():
    """Real measured value after the colour split; occlusion and substitutions
    make an exact six unrealistic."""
    v = assess(_stats(team=7.0))
    assert not v.fatal
    assert v.findings == []


# --------------------------------------------------------------------------
# boundaries
# --------------------------------------------------------------------------

def test_too_few_players_is_fatal_not_a_warning():
    """Fewer than three players cannot fill six slots, so spacing/centroid
    features would describe players who are not there."""
    v = assess(_stats(team=MIN_TEAM_MEDIAN - 1))
    assert v.fatal
    assert any(f.stage == "team split" for f in v.findings)


def test_both_teams_leaking_is_a_warning_not_fatal():
    """The pre-fix geometric split passed ten mixed-team players. The output is
    wrong but recoverable, and the user may have deliberately set --team-side
    all, so warn rather than abort."""
    v = assess(_stats(team=MAX_TEAM_MEDIAN + 1))
    assert not v.fatal
    assert any(f.level == WARN and f.stage == "team split" for f in v.findings)
    assert "colour" in v.warnings[0].remedy


def test_slightly_off_expected_size_warns_only():
    v = assess(_stats(team=9.0))
    assert not v.fatal
    assert v.warnings


def test_low_ball_recall_warns_and_names_the_affected_rules():
    """Two of the four tactical rules depend on the ball; at low recall their
    labels are noise, and the report must say so rather than imply confidence."""
    v = assess(_stats(ball=0.05))
    assert not v.fatal
    w = [f for f in v.warnings if f.stage == "ball"]
    assert w and "Delayed-Support" in w[0].message


def test_good_ball_recall_produces_no_ball_warning():
    v = assess(_stats(ball=0.77))
    assert not any(f.stage == "ball" for f in v.findings)


# --------------------------------------------------------------------------
# short-circuiting and reporting
# --------------------------------------------------------------------------

def test_detection_failure_short_circuits_downstream_noise():
    """When the detector is blind, every downstream stage is trivially zero;
    reporting three fatals would bury the one that matters."""
    v = assess(_stats(raw=0.0, masked=0.0, team=0.0, ball=0.0))
    assert len(v.findings) == 1


def test_every_finding_carries_an_actionable_remedy():
    for stats in (_stats(raw=0.0), _stats(raw=20, masked=0.0),
                  _stats(team=0.0), _stats(team=12.0), _stats(ball=0.0)):
        for f in assess(stats).findings:
            assert f.remedy and len(f.remedy) > 20
            assert f.level in (FATAL, WARN)


def test_render_marks_fatal_findings():
    assert "FATAL" in assess(_stats(raw=0.0)).render()


def test_summarise_reports_every_measured_stage():
    out = summarise(_stats())
    for token in ("frames sampled", "detector", "court mask", "team split",
                  "ball recall"):
        assert token in out


def test_summarise_omits_absent_measurements():
    out = summarise({"raw_person_median": 20.0})
    assert "ball recall" not in out


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-q"]))
