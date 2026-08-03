"""Regression test for the half-court auto-calibration bug.

Real failure (videoplayback (4).mp4, frame 11290): the auto-detected quad is
the NEAR half-court (top edge = net floor line), but it was mapped onto the
FULL court plane - so front-row players at the net mapped to court_y~334
("opponent side") and the bottom-team filter deleted them; in high formations
the entire team vanished and every window became TRANSITION/DEAD BALL.

Uses the exact quad detected from that real frame. cv2+numpy only."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from court import NET_Y_CM, CourtCalibrator  # noqa: E402

# quad measured by detect_court_quad on the real frame (TL, TR, BR, BL)
REAL_QUAD = np.array([[296, 293], [925, 285], [1165, 696], [117, 685]],
                     dtype=np.float32)
MARGIN = 30.0                                  # pipeline default net margin


def _calibrator_with_auto_quad():
    cal = CourtCalibrator(1280, 720, auto=True)
    cal.set_auto_quad(REAL_QUAD.copy())
    return cal


def test_quad_top_edge_is_the_net():
    cal = _calibrator_with_auto_quad()
    for fx in (0.3, 0.5, 0.7):
        px = 296 + fx * (925 - 296)
        py = 293 + fx * (285 - 293)
        _, cy = cal.pixel_to_court(px, py)
        assert abs(cy - NET_Y_CM) < 30, cy     # top edge maps to ~450 (net)


def test_front_row_players_stay_on_our_side():
    """The exact failing case: feet at pixel y~396 (attack zone) must map
    INSIDE our half and pass the bottom-side filter."""
    cal = _calibrator_with_auto_quad()
    kept = 0
    for px, py in [(384, 396), (576, 396), (768, 396), (896, 396),
                   (384, 468), (768, 468), (576, 540), (768, 612)]:
        _, cy = cal.pixel_to_court(px, py)
        assert NET_Y_CM <= cy <= 900 + 1, (px, py, cy)
        if cy >= NET_Y_CM + MARGIN:
            kept += 1
    assert kept == 8                            # nobody deleted any more


def test_far_side_players_still_filtered_out():
    cal = _calibrator_with_auto_quad()
    # feet ABOVE the quad's top edge (far side of the net, py < 285)
    _, cy = cal.pixel_to_court(600.0, 250.0)
    assert cy < NET_Y_CM + MARGIN               # correctly not "our side"


def test_manual_corners_keep_full_court_semantics():
    corners = [(100, 100), (1180, 100), (1180, 620), (100, 620)]
    cal = CourtCalibrator(1280, 720, manual_corners=corners)
    _, cy_top = cal.pixel_to_court(640.0, 100.0)
    _, cy_bot = cal.pixel_to_court(640.0, 620.0)
    assert abs(cy_top - 0.0) < 1 and abs(cy_bot - 900.0) < 1


def test_force_linear_masks_but_keeps_linear_coords():
    """Serving a linear-coordinate-trained model: the quad must still REJECT
    off-court people (the coach who stole slot P2 in the real failure) while
    coordinates remain on the training-identical linear mapping."""
    cal = CourtCalibrator(1280, 720, auto=True, force_linear=True)
    cal.set_auto_quad(REAL_QUAD.copy())
    # coach at the right sideline apron (x~1240) - outside the quad
    assert not cal.in_court(1240.0, 500.0)
    # on-court player - inside the quad
    assert cal.in_court(640.0, 500.0)
    # coordinates = LINEAR mapping (py/720*900), NOT the homography
    _, cy = cal.pixel_to_court(640.0, 500.0)
    assert abs(cy - 500.0 / 720.0 * 900.0) < 1e-6


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print("  PASS  " + fn.__name__)
    print("\n{}/{} tests passed.".format(len(fns), len(fns)))


if __name__ == "__main__":
    _run_all()
