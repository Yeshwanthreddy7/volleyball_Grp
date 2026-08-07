"""
court.py – Coordinate calibration, dynamic homography, camera-motion
compensation and the court mask (spec §3).

Why a static pixel→cm mapping is wrong for broadcast video
----------------------------------------------------------
Broadcast cameras pan, tilt and zoom.  With a fixed homography, pure camera
motion is indistinguishable from player motion: a 2° pan adds a phantom
~100 cm/s to every track.  This module keeps the calibration LIVE:

  1. FRAME OF REFERENCE.  All positions are expressed on a flat, normalised
     1800 cm × 900 cm court plane (both half-courts; net at y = 450 cm).
     A 3×3 homography H maps pixels (u, v) to that plane:

         [x', y', w']ᵀ = H · [u, v, 1]ᵀ ;   X = x'/w' ,  Y = y'/w'

     H is exact for points ON the floor plane – which is why player position
     is taken at the FEET (features.foot_point), never the box centroid.

  2. HOW THE CORNER VALUES ARE OBTAINED.
     (a) Manual: the operator supplies the 4 pixel corners of the court
         (TL TR BR BL) once, e.g. read off a paused frame.  Best accuracy.
     (b) Automatic (--auto-court): every `refresh_every` frames the court is
         segmented by colour + contour analysis and reduced to a convex
         quadrilateral (approxPolyDP over the dominant floor region, an
         efficient equivalent of intersecting probabilistic-Hough boundary
         lines).  Detections must pass sanity checks (convexity, area ratio,
         landscape aspect) before they replace the current corners – a failed
         detection NEVER destroys a working calibration.

  3. CAMERA MOTION COMPENSATION (--cmc).  Between refreshes, sparse optical
     flow (Shi–Tomasi corners + pyramidal Lucas-Kanade) is tracked from frame
     to frame and a partial-affine camera model is fitted with RANSAC (player
     motion is rejected as outliers because the majority of features sit on
     the static background).  The affine warp is applied to the stored court
     corners, and H is recomputed – so the pixel→cm mapping follows the pan.

  4. BACKGROUND FALSE-POSITIVE ELIMINATION.  A binary polygon mask built from
     the corners drops every detection whose FOOT POINT lies outside the
     court (+margin): referees, bench, coaches and crowd never reach the
     tracker, protecting the 6-player matrix.
"""

from __future__ import annotations

import sys

import cv2
import numpy as np

COURT_W_CM = 1800.0
COURT_H_CM = 900.0
NET_Y_CM = COURT_H_CM / 2.0

# Destination quad on the normalised court plane (TL TR BR BL).
_DST = np.array(
    [[0, 0], [COURT_W_CM, 0], [COURT_W_CM, COURT_H_CM], [0, COURT_H_CM]],
    dtype=np.float32,
)


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 arbitrary points TL, TR, BR, BL (sum/diff heuristic)."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()   # y - x
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.stack([tl, tr, br, bl])


_HALF_DST = np.array(
    [[0, NET_Y_CM], [COURT_W_CM, NET_Y_CM],
     [COURT_W_CM, COURT_H_CM], [0, COURT_H_CM]], dtype=np.float32)


def build_homography(corners: np.ndarray, dst: np.ndarray | None = None) -> np.ndarray | None:
    """3×3 homography mapping the 4 pixel corners (TL TR BR BL) to the court.

    dst=None maps to the FULL court plane (manual corner calibration).
    dst=_HALF_DST maps to the NEAR half (net at the quad's top edge) - used
    for auto-detected quads, because colour segmentation of broadcast footage
    finds the NEAR half-court (the far half is compressed and net-occluded).
    Mapping a half-court quad onto the full plane put the real net at y=0 and
    the attack zone at y<450, so the team-side filter deleted the entire
    front row (and in high formations, the whole team) as "opponents"."""
    src = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    H, _ = cv2.findHomography(src, _DST if dst is None else dst)
    return H


def detect_court_quad(frame: np.ndarray) -> np.ndarray | None:
    """
    Segment the dominant playing surface and reduce it to a 4-corner convex
    quadrilateral.  Returns (4,2) corners in TL,TR,BR,BL order, or None.

    Method: sample the court colour from the lower-central region (where the
    floor almost always is in a broadcast frame), threshold in HSV, clean with
    morphology, take the largest contour and approximate its convex hull down
    to 4 vertices.  All rejects return None – caller keeps its previous
    calibration.
    """
    h, w = frame.shape[:2]
    scale = 480.0 / max(w, 1)
    small = cv2.resize(frame, (int(w * scale), int(h * scale)))
    sh, sw = small.shape[:2]

    hsv = cv2.cvtColor(cv2.GaussianBlur(small, (5, 5), 0), cv2.COLOR_BGR2HSV)

    # Sample floor colour from the lower-central patch.
    patch = hsv[int(sh * 0.55): int(sh * 0.9), int(sw * 0.3): int(sw * 0.7)]
    if patch.size == 0:
        return None
    med = np.median(patch.reshape(-1, 3), axis=0)
    lo = np.array([max(med[0] - 12, 0), max(med[1] - 70, 0), max(med[2] - 70, 0)])
    hi = np.array([min(med[0] + 12, 179), min(med[1] + 70, 255), min(med[2] + 70, 255)])
    mask = cv2.inRange(hsv, lo.astype(np.uint8), hi.astype(np.uint8))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < 0.08 * sh * sw or area > 0.95 * sh * sw:
        return None

    hull = cv2.convexHull(cnt)
    quad = None
    for eps_frac in (0.02, 0.03, 0.05, 0.08):
        approx = cv2.approxPolyDP(hull, eps_frac * cv2.arcLength(hull, True), True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype(np.float32)
            break
    if quad is None:
        return None
    if not cv2.isContourConvex(quad.astype(np.int32)):
        return None

    corners = order_corners(quad) / scale     # back to full resolution

    # Sanity: landscape shape (court is wider than tall on screen).
    width_px = np.linalg.norm(corners[1] - corners[0])
    height_px = np.linalg.norm(corners[3] - corners[0])
    if width_px < height_px:
        return None
    return corners


class CameraMotionEstimator:
    """Frame-to-frame partial-affine camera model from sparse optical flow."""

    def __init__(self, max_features: int = 400) -> None:
        self._prev_gray: np.ndarray | None = None
        self._max_features = max_features

    def reset(self) -> None:
        self._prev_gray = None

    def estimate(self, frame: np.ndarray) -> np.ndarray | None:
        """Return a 2×3 affine A such that  p_curr ≈ A · [p_prev, 1]ᵀ,
        or None when motion cannot be estimated (first frame, no features)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=0.5, fy=0.5)

        A = None
        if self._prev_gray is not None:
            p0 = cv2.goodFeaturesToTrack(
                self._prev_gray, maxCorners=self._max_features,
                qualityLevel=0.01, minDistance=12, blockSize=7,
            )
            if p0 is not None and len(p0) >= 12:
                p1, st, _ = cv2.calcOpticalFlowPyrLK(
                    self._prev_gray, gray, p0, None,
                    winSize=(21, 21), maxLevel=3,
                )
                if p1 is not None:
                    good = st.ravel() == 1
                    if good.sum() >= 12:
                        A_half, _ = cv2.estimateAffinePartial2D(
                            p0[good], p1[good], method=cv2.RANSAC,
                            ransacReprojThreshold=3.0,
                        )
                        if A_half is not None:
                            # rescale translation from half-res to full-res
                            A = A_half.copy()
                            A[:, 2] *= 2.0
        self._prev_gray = gray
        return A


class CourtCalibrator:
    """
    Stateful pixel→court calibration for one video.

    Modes (all combinable):
      * manual corners             – fixed H (classic behaviour)
      * auto=True                  – periodic court re-detection every
                                     `refresh_every` frames
      * cmc=True                   – per-frame camera-motion warp of the
                                     corners between (re)detections
      * neither corners nor auto   – linear fallback scaling, mask disabled

    Call `update(frame)` once per frame BEFORE converting coordinates.
    """

    def __init__(
        self,
        frame_w: int,
        frame_h: int,
        manual_corners: np.ndarray | list | None = None,
        auto: bool = False,
        cmc: bool = False,
        refresh_every: int = 5,
        mask_margin_px: float = 12.0,
        force_linear: bool = False,
        manual_half: bool = False,
    ) -> None:
        # force_linear: use the detected quad ONLY as an in/out-of-court MASK
        # (drop coaches/refs/bench) while coordinates stay on the linear
        # pixel->court fallback. This matches the training-feature geometry
        # (CSVs were extracted with linear coords) while still cleaning the
        # detection population on full broadcast video.
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.auto = auto
        self.cmc = cmc
        self.refresh_every = max(int(refresh_every), 1)
        self.mask_margin_px = mask_margin_px
        self.force_linear = bool(force_linear)

        self.corners: np.ndarray | None = (
            order_corners(np.asarray(manual_corners, dtype=np.float32))
            if manual_corners is not None else None
        )
        # Manual corners normally describe the FULL court; auto-detected quads
        # describe the NEAR half (see build_homography docstring). Pass
        # manual_half=True when the 4 manual points ALSO only span the near
        # half (net -> near baseline) - e.g. because the far baseline isn't
        # reliably visible from this camera angle - so they get the same
        # _HALF_DST mapping an auto-detected quad would. Getting this wrong
        # silently shifts the internal net-line reference and can flip which
        # team gets treated as "near".
        self._dst: np.ndarray | None = (
            _HALF_DST if (manual_corners is None or manual_half) else None
        )
        self.H: np.ndarray | None = (
            build_homography(self.corners, dst=self._dst) if self.corners is not None else None
        )
        self._cme = CameraMotionEstimator() if cmc else None
        self._frame_idx = 0
        self.n_auto_success = 0
        self.n_auto_fail = 0

    # ------------------------------------------------------------------
    @property
    def calibrated(self) -> bool:
        return self.H is not None

    def update(self, frame: np.ndarray) -> None:
        """Advance calibration by one frame (CMC warp + periodic re-detect)."""
        self._frame_idx += 1

        # 1. Camera-motion compensation: warp stored corners by camera affine.
        if self._cme is not None:
            A = self._cme.estimate(frame)
            if A is not None and self.corners is not None:
                pts = np.hstack([self.corners, np.ones((4, 1), dtype=np.float32)])
                self.corners = (A @ pts.T).T.astype(np.float32)
                self.H = build_homography(self.corners, dst=self._dst)

        # 2. Periodic automatic (re)detection.
        if self.auto and (self._frame_idx % self.refresh_every == 1
                          or self.corners is None):
            quad = detect_court_quad(frame)
            if quad is not None:
                self.set_auto_quad(quad)
                self.n_auto_success += 1
            else:
                self.n_auto_fail += 1   # keep previous calibration

    # ------------------------------------------------------------------
    def set_auto_quad(self, quad: np.ndarray) -> None:
        """Adopt an auto-detected NEAR-HALF-court quad (net = top edge)."""
        self.corners = quad
        self._dst = _HALF_DST
        self.H = build_homography(quad, dst=_HALF_DST)

    # ------------------------------------------------------------------
    def pixel_to_court(self, px: float, py: float) -> tuple[float, float]:
        """Map one pixel to court cm (homography, or linear fallback)."""
        if self.H is not None and not self.force_linear:
            pt = np.array([[[px, py]]], dtype=np.float32)
            out = cv2.perspectiveTransform(pt, self.H)
            return float(out[0, 0, 0]), float(out[0, 0, 1])
        return (
            px / max(self.frame_w, 1) * COURT_W_CM,
            py / max(self.frame_h, 1) * COURT_H_CM,
        )

    # ------------------------------------------------------------------
    def in_court(self, px: float, py: float) -> bool:
        """True when a pixel lies inside the court polygon (+margin).
        Without a calibration polygon everything passes (no-op mask)."""
        if self.corners is None:
            return True
        poly = self.corners.astype(np.float32)
        d = cv2.pointPolygonTest(poly, (float(px), float(py)), True)
        return d >= -self.mask_margin_px

    def filter_detections(
        self, xyxy: np.ndarray, conf: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Drop detections whose FOOT POINT is outside the court polygon –
        referees, bench players, coaches, crowd (spec §3C)."""
        if self.corners is None or len(xyxy) == 0:
            return xyxy, conf
        keep = [
            i for i, (x1, y1, x2, y2) in enumerate(xyxy)
            if self.in_court((x1 + x2) / 2.0, y2)
        ]
        if not keep:
            return (np.empty((0, 4), dtype=xyxy.dtype),
                    np.empty((0,), dtype=conf.dtype))
        idx = np.asarray(keep, dtype=int)
        return xyxy[idx], conf[idx]

    # ------------------------------------------------------------------
    def court_mask(self) -> np.ndarray | None:
        """Binary uint8 mask (255 inside court) or None when uncalibrated."""
        if self.corners is None:
            return None
        mask = np.zeros((self.frame_h, self.frame_w), dtype=np.uint8)
        cv2.fillPoly(mask, [self.corners.astype(np.int32)], 255)
        return mask

    def draw(self, frame: np.ndarray) -> None:
        """Overlay the current court polygon on a frame (in place)."""
        if self.corners is None:
            return
        cv2.polylines(
            frame, [self.corners.astype(np.int32)], isClosed=True,
            color=(0, 220, 220), thickness=2, lineType=cv2.LINE_AA,
        )


__all__ = [
    "COURT_W_CM", "COURT_H_CM", "NET_Y_CM",
    "order_corners", "build_homography", "detect_court_quad",
    "CameraMotionEstimator", "CourtCalibrator",
]
