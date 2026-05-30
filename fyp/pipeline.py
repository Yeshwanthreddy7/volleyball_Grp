"""
pipeline.py – End-to-end volleyball analytics pipeline.

Implements the complete processing chain from a raw match video to a
printed tactical coordination report:

  Match Video (.mp4)
        ↓
  Frame Extraction
        ↓
  Player Detection  (YOLOv8)
        ↓
  Player Tracking   (ByteTrack via supervision)
        ↓
  Pose / Position Extraction  (bounding-box centroids → court coordinates)
        ↓
  Feature Engineering  (velocity, spacing, centroid, sync_score)
        ↓
  Sequence Formation  (non-overlapping 29-frame windows)
        ↓
  Mamba Model  (Sequence Learning / Coordination Classification)
        ↓
  Final Tactical Report

Usage
-----
  python pipeline.py <video> <checkpoint> [options]

Example
-------
  python pipeline.py match.mp4 mamba_checkpoint.pt \\
      --output-video annotated.mp4 \\
      --output-csv   predictions.csv

Court coordinate system
-----------------------
  Default: pixel coordinates are linearly scaled to 1800 × 900 cm
  (1800 cm wide, 900 cm tall, net at Y = 450 cm for top-down view).

  For accurate court mapping supply --court-corners with the four pixel
  coordinates of the court boundary corners (TL TR BR BL) and the script
  will compute a perspective homography.

  Example:
    --court-corners 42,18 1238,18 1238,702 42,702
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict, deque

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:  # pragma: no cover
    HAS_ULTRALYTICS = False

try:
    import supervision as sv
    HAS_SUPERVISION = True
except ImportError:  # pragma: no cover
    HAS_SUPERVISION = False

from mamba_model import FEATURE_COLS, INPUT_DIM, LABEL_NAMES, MambaClassifier
from label_clips import LABEL_TO_INDEX, generate_report, _compute_sync_score, label_clip


# ---------------------------------------------------------------------------
# Pipeline constants
# ---------------------------------------------------------------------------

PERSON_CLASS_ID = 0        # COCO class 0 = person
SPORTS_BALL_CLASS_ID = 32  # COCO class 32 = sports ball (standard COCO model)
SEQ_LEN = 29               # frames per sequence (matches training window)
N_PLAYERS = 6              # players tracked per sequence

COURT_W_CM = 1800.0        # court width  (cm)
COURT_H_CM = 900.0         # court height (cm)
NET_Y_CM = COURT_H_CM / 2.0

# Visual style for annotated output video
TRACK_COLOR = (0, 255, 0)       # green – player bounding boxes (default / no label yet)
BALL_COLOR = (0, 165, 255)      # orange – ball marker
TEXT_COLOR = (255, 255, 255)    # white – overlaid text
LABEL_BG_COLOR = (30, 30, 30)  # dark – label banner background (default)
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Label-specific box / banner colors (BGR) for tactical classification visualization
#   🟠 1 – Coordinated Attack  → orange
#   🔵 2 – Coordinated Defense → blue
#   🩵 3 – Delayed Support     → cyan
#   🔴 4 – Spacing Breakdown   → red
LABEL_COLORS: dict[str, tuple[int, int, int]] = {
    "Coordinated Attack":  (0, 165, 255),   # BGR: orange
    "Coordinated Defense": (255, 0, 0),      # BGR: blue
    "Delayed Support":     (255, 255, 0),    # BGR: cyan
    "Spacing Breakdown":   (0, 0, 255),      # BGR: red
}
FOURCC = cv2.VideoWriter_fourcc(*"mp4v")

# Number of historical ball positions kept for the trail overlay.
BALL_TRAIL_LEN = 45

# Maximum consecutive frames the BallTracker will predict ball position
# (constant-velocity model) without a new YOLO detection.
BALL_MAX_MISS_FRAMES = 12

# Simplified phase names displayed prominently in the annotation banner.
# Internal label name → user-facing display string.
DISPLAY_NAMES: dict[str, str] = {
    "Coordinated Attack":  "ATTACKING",
    "Coordinated Defense": "DEFENDING",
    "Delayed Support":     "DELAYED SUPPORT",
    "Spacing Breakdown":   "SPACING BREAKDOWN",
}


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def _load_mamba(
    path: str, device: torch.device
) -> tuple[MambaClassifier, torch.Tensor, torch.Tensor]:
    """Load a Mamba checkpoint; return (model, norm_mean, norm_std)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})
    model = MambaClassifier(
        input_dim=saved_args.get("input_dim", INPUT_DIM),
        d_model=saved_args.get("d_model", 64),
        n_layers=saved_args.get("n_layers", 4),
        d_state=saved_args.get("d_state", 16),
        d_conv=saved_args.get("d_conv", 4),
        num_classes=len(LABEL_NAMES),
        dropout=0.0,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    mean = ckpt["norm_mean"].to(device)
    std = ckpt["norm_std"].to(device)
    return model, mean, std


# ---------------------------------------------------------------------------
# Ball tracking (constant-velocity model with EMA velocity smoothing)
# ---------------------------------------------------------------------------

class BallTracker:
    """
    Lightweight ball tracker that bridges gaps in YOLO ball detections.

    YOLOv8 (especially the generic COCO model) frequently misses a fast-moving
    volleyball for several consecutive frames due to motion blur, small
    apparent size, or occlusion.  This class maintains an estimated ball state
    (position + velocity) and predicts the ball's position during missed
    frames using a constant-velocity kinematic model.

    Detection vs Prediction
    -----------------------
    • When YOLO finds the ball → state is updated with the measured centre
      and velocity is refined with an exponential moving average (EMA).
    • When YOLO misses the ball → position is advanced by the last velocity
      estimate.  After ``BALL_MAX_MISS_FRAMES`` consecutive misses the tracker
      gives up (returns None) so stale predictions don't persist indefinitely.

    Parameters
    ----------
    max_miss_frames : int
        Maximum consecutive frames the tracker will predict without a fresh
        YOLO detection.  Default: ``BALL_MAX_MISS_FRAMES`` (12 frames).
    velocity_alpha  : float
        EMA weight applied to each new velocity measurement.
        Higher values → more responsive but noisier.  Default: 0.55.

    Usage
    -----
    ::

        tracker = BallTracker()
        for frame in video_frames:
            detected_center = yolo_detect_ball(frame)   # np.ndarray or None
            ball_pos, is_predicted = tracker.update(detected_center)
            # ball_pos: (2,) pixel coords or None (ball lost)
            # is_predicted: True when position is an extrapolation
    """

    def __init__(
        self,
        max_miss_frames: int = BALL_MAX_MISS_FRAMES,
        velocity_alpha: float = 0.55,
    ) -> None:
        self._max_miss = max_miss_frames
        self._alpha = velocity_alpha
        self._pos: np.ndarray | None = None   # last known pixel (x, y)
        self._vel: np.ndarray = np.zeros(2)   # estimated velocity (dx, dy) in pixels/frame
        self._miss: int = 0                   # consecutive missed-detection frames

    def update(
        self, detected_center: np.ndarray | None
    ) -> tuple[np.ndarray | None, bool]:
        """
        Advance the tracker by one frame.

        Parameters
        ----------
        detected_center : (2,) pixel position from YOLO, or None if not found.

        Returns
        -------
        position     : (2,) pixel position estimate, or None when ball is lost.
        is_predicted : True when the returned position is an extrapolation;
                       False when it comes directly from YOLO.
        """
        if detected_center is not None:
            # Ball detected – update velocity with EMA and reset miss counter
            if self._pos is not None:
                raw_vel = detected_center - self._pos
                self._vel = self._alpha * raw_vel + (1.0 - self._alpha) * self._vel
            self._pos = detected_center.astype(float).copy()
            self._miss = 0
            return self._pos.copy(), False

        # Ball not detected
        self._miss += 1
        if self._miss > self._max_miss or self._pos is None:
            # Prediction horizon exceeded or ball was never seen → give up
            return None, False

        # Predict next position using last known velocity
        self._pos = self._pos + self._vel
        return self._pos.copy(), True

    def reset(self) -> None:
        """Reset internal state (use between video clips)."""
        self._pos = None
        self._vel = np.zeros(2)
        self._miss = 0


# ---------------------------------------------------------------------------
# Homography helper
# ---------------------------------------------------------------------------

def _build_homography(corners: list[tuple[float, float]]) -> np.ndarray:
    """
    Compute a perspective homography that maps four pixel corner points
    (TL, TR, BR, BL order) to the 1800 × 900 cm court coordinate system.

    Parameters
    ----------
    corners : [(x0,y0), (x1,y1), (x2,y2), (x3,y3)]  pixel coordinates
              in top-left → top-right → bottom-right → bottom-left order.

    Returns
    -------
    H : (3,3) homography matrix
    """
    src = np.array(corners, dtype=np.float32)
    # Destination: the four court corners in cm
    dst = np.array(
        [[0, 0], [COURT_W_CM, 0], [COURT_W_CM, COURT_H_CM], [0, COURT_H_CM]],
        dtype=np.float32,
    )
    H, _ = cv2.findHomography(src, dst)
    return H


def _pixel_to_court(
    px: float,
    py: float,
    frame_w: int,
    frame_h: int,
    H: np.ndarray | None,
) -> tuple[float, float]:
    """Convert a single pixel position to court coordinates (cm)."""
    if H is not None:
        pt = np.array([[[px, py]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, H)
        return float(out[0, 0, 0]), float(out[0, 0, 1])
    # Fallback: linear scaling
    return px / frame_w * COURT_W_CM, py / frame_h * COURT_H_CM


# ---------------------------------------------------------------------------
# Detection (YOLOv8)
# ---------------------------------------------------------------------------

def _detect(
    frame: np.ndarray,
    yolo: "YOLO",
    conf_threshold: float = 0.35,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Run YOLOv8 on *frame*.

    Returns
    -------
    person_xyxy : (N, 4) bounding boxes for detected persons
    person_conf : (N,) confidence scores
    ball_center : (2,) pixel position of the highest-confidence ball, or None
    """
    results = yolo(frame, verbose=False, conf=conf_threshold)[0]
    boxes = results.boxes

    cls_ids = boxes.cls.int().cpu().numpy()
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()

    # Persons
    person_mask = cls_ids == PERSON_CLASS_ID
    person_xyxy = xyxy[person_mask]
    person_conf = conf[person_mask]

    # Sports ball (class 32) – may be absent in sport-specific YOLOv8 weights
    ball_center: np.ndarray | None = None
    ball_mask = cls_ids == SPORTS_BALL_CLASS_ID
    if ball_mask.any():
        ball_boxes = xyxy[ball_mask]
        ball_confs = conf[ball_mask]
        best = int(np.argmax(ball_confs))
        x1, y1, x2, y2 = ball_boxes[best]
        ball_center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=float)

    return person_xyxy, person_conf, ball_center


# ---------------------------------------------------------------------------
# Tracking (ByteTrack)
# ---------------------------------------------------------------------------

def _track(
    person_xyxy: np.ndarray,
    person_conf: np.ndarray,
    tracker: "sv.ByteTrack",
) -> "sv.Detections":
    """Update ByteTrack and return annotated Detections with tracker_id."""
    if len(person_xyxy) == 0:
        return sv.Detections.empty()
    detections = sv.Detections(
        xyxy=person_xyxy,
        confidence=person_conf,
        class_id=np.zeros(len(person_xyxy), dtype=int),
    )
    return tracker.update_with_detections(detections)


def _filter_players_by_team_side(
    person_xyxy: np.ndarray,
    person_conf: np.ndarray,
    frame_w: int,
    frame_h: int,
    H: np.ndarray | None,
    team_side: str,
    net_margin_cm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Keep only player detections on the requested side of the net.

    team_side values:
      - "bottom": keep players with court_y >= NET_Y_CM + net_margin_cm
      - "top"   : keep players with court_y <= NET_Y_CM - net_margin_cm
      - "all"   : keep every detected player
    """
    if team_side == "all" or len(person_xyxy) == 0:
        return person_xyxy, person_conf

    keep_indices: list[int] = []
    for i, (x1, y1, x2, y2) in enumerate(person_xyxy):
        px = float((x1 + x2) / 2.0)
        py = float((y1 + y2) / 2.0)
        _, court_y = _pixel_to_court(px, py, frame_w, frame_h, H)

        if team_side == "bottom":
            on_side = court_y >= (NET_Y_CM + net_margin_cm)
        else:  # team_side == "top"
            on_side = court_y <= (NET_Y_CM - net_margin_cm)

        if on_side:
            keep_indices.append(i)

    if not keep_indices:
        return (
            np.empty((0, 4), dtype=person_xyxy.dtype),
            np.empty((0,), dtype=person_conf.dtype),
        )

    idx = np.array(keep_indices, dtype=int)
    return person_xyxy[idx], person_conf[idx]


# ---------------------------------------------------------------------------
# Position extraction (Pose / Position Extraction step)
# ---------------------------------------------------------------------------

def _extract_positions(
    tracks: "sv.Detections",
    ball_center: np.ndarray | None,
    frame_w: int,
    frame_h: int,
    H: np.ndarray | None,
    track_age: dict[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert ByteTrack bounding-box centroids to court coordinates.

    Always returns exactly N_PLAYERS player positions (padded with zeros
    when fewer tracks are available).

    Parameters
    ----------
    tracks     : ByteTrack output Detections
    ball_center: pixel (x, y) of ball, or None
    track_age  : mutable dict accumulating per-ID frame counts (for
                 stability-based player slot assignment)

    Returns
    -------
    player_positions : (6, 2) court cm coordinates
    ball_pos         : (2,) court cm coordinates  (0,0 if ball unknown)
    """
    # Accumulate track ages so older (more stable) tracks get priority slots
    if tracks.tracker_id is not None:
        for tid in tracks.tracker_id:
            track_age[int(tid)] = track_age.get(int(tid), 0) + 1

    # Collect bounding-box centres per active track ID
    centers: dict[int, tuple[float, float]] = {}
    if tracks.tracker_id is not None and len(tracks.xyxy) > 0:
        for tid, (x1, y1, x2, y2) in zip(tracks.tracker_id, tracks.xyxy):
            centers[int(tid)] = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    # Sort by descending age → assign most stable N_PLAYERS to fixed slots
    chosen = sorted(centers.keys(), key=lambda t: track_age.get(t, 0), reverse=True)
    chosen = chosen[:N_PLAYERS]

    player_positions = np.zeros((N_PLAYERS, 2), dtype=float)
    for slot, tid in enumerate(chosen):
        px, py = centers[tid]
        cx, cy = _pixel_to_court(px, py, frame_w, frame_h, H)
        player_positions[slot] = [cx, cy]

    # Ball
    if ball_center is not None:
        bx, by = _pixel_to_court(
            float(ball_center[0]), float(ball_center[1]), frame_w, frame_h, H
        )
    else:
        bx, by = 0.0, 0.0

    return player_positions, np.array([bx, by], dtype=float)


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

def _compute_frame_features(
    player_positions: np.ndarray,  # (6, 2)
    ball_pos: np.ndarray,          # (2,)
) -> np.ndarray:
    """
    Build a 14-element feature row matching the model's FEATURE_COLS:
      [ball_x, ball_y, p1_x, p1_y, p2_x, p2_y, …, p6_x, p6_y]

    Parameters
    ----------
    player_positions : (6, 2) court cm coordinates
    ball_pos         : (2,) court cm coordinates

    Returns
    -------
    row : (14,) float32 feature vector
    """
    row = np.empty(INPUT_DIM, dtype=np.float32)
    row[0] = ball_pos[0]
    row[1] = ball_pos[1]
    for i in range(N_PLAYERS):
        row[2 + i * 2] = player_positions[i, 0]
        row[2 + i * 2 + 1] = player_positions[i, 1]
    return row


def _sequence_metrics(seq: np.ndarray) -> dict[str, float]:
    """
    Compute descriptive metrics for a 29-frame sequence.

    Returns a dict with:
      sync_score     – movement synchronization (mean cosine similarity)
      mean_spacing   – average nearest-neighbour spacing (cm)
      centroid_vel   – mean team centroid velocity (cm/frame)
    """
    # Player positions: columns 2..13, reshape to (29, 6, 2)
    positions = seq[:, 2:].reshape(len(seq), N_PLAYERS, 2)

    # Sync score
    sync = _compute_sync_score(positions)

    # Mean nearest-neighbour spacing per frame (skip zero-padded slots).
    # Vectorised: compute all pairwise distances per frame with NumPy broadcasting.
    spacing_samples: list[float] = []
    for pos_frame in positions:
        occupied = pos_frame[~np.all(pos_frame == 0, axis=1)]  # (k, 2)
        if len(occupied) < 2:
            continue
        diff = occupied[:, np.newaxis, :] - occupied[np.newaxis, :, :]  # (k, k, 2)
        dists = np.linalg.norm(diff, axis=-1)                           # (k, k)
        np.fill_diagonal(dists, np.inf)
        spacing_samples.extend(dists.min(axis=-1).tolist())
    mean_spacing = float(np.mean(spacing_samples)) if spacing_samples else 0.0

    # Team centroid velocity (frame-to-frame)
    centroid = positions.mean(axis=1)  # (29, 2)
    centroid_vel = float(np.linalg.norm(np.diff(centroid, axis=0), axis=1).mean())

    return {
        "sync_score": round(sync, 4),
        "mean_spacing_cm": round(mean_spacing, 1),
        "centroid_vel_cm_per_frame": round(centroid_vel, 2),
    }


# ---------------------------------------------------------------------------
# Mamba inference (Sequence Learning + Classification)
# ---------------------------------------------------------------------------

def _classify_sequence(
    seq: np.ndarray,            # (29, 14) float32
    model: MambaClassifier,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> tuple[str, int, float, torch.Tensor]:
    """
    Classify a 29-frame feature sequence with the trained Mamba model.

    Returns
    -------
    label      : predicted label string
    label_idx  : 1-based numeric index
    confidence : softmax probability of predicted class
    probs      : (num_classes,) probability tensor
    """
    t = torch.from_numpy(seq).to(device)   # (29, 14)
    t = (t - mean) / std                   # z-score normalisation
    with torch.no_grad():
        logits = model(t.unsqueeze(0))     # (1, num_classes)
        probs = F.softmax(logits, dim=-1)[0]
    pred_idx = int(probs.argmax().item())
    label = LABEL_NAMES[pred_idx]
    conf = float(probs[pred_idx].item())
    numeric_idx = LABEL_TO_INDEX.get(label, 0)
    return label, numeric_idx, conf, probs


def _classify_sequence_rule_based(seq: np.ndarray) -> tuple[str, int, float, None]:
    """
    Classify a 29-frame sequence using the rule-based engine from label_clips.

    Converts the numpy feature array to a DataFrame matching the format
    expected by label_clip() and applies the four labeling rules.

    Returns
    -------
    label      : predicted label string
    label_idx  : 1-based numeric index
    confidence : 1.0 (rule-based decisions are deterministic)
    probs      : None  (no probability vector in rule-based mode)
    """
    df = pd.DataFrame(seq, columns=FEATURE_COLS)
    df.insert(0, "frame_id", range(1, len(df) + 1))
    label = label_clip(df)
    if label == "Unclassified":
        # Default to Spacing Breakdown when no rule fires (rare edge case)
        label = "Spacing Breakdown"
    numeric_idx = LABEL_TO_INDEX.get(label, 0)
    return label, numeric_idx, 1.0, None


# ---------------------------------------------------------------------------
# Frame annotation
# ---------------------------------------------------------------------------

def _annotate_frame(
    frame: np.ndarray,
    tracks: "sv.Detections",
    ball_center: np.ndarray | None,
    ball_is_predicted: bool,
    ball_trail: "deque[tuple[int, int, bool]]",
    current_label: str,
    current_conf: float,
    frame_idx: int,
    fps: float,
) -> np.ndarray:
    """
    Draw tracking boxes, ball trail + marker, frame info, and current label.

    Ball trail
    ----------
    Up to BALL_TRAIL_LEN past ball positions are stored as (x, y, is_predicted)
    tuples and drawn as small fading circles:
      • Detected positions  : orange circles, fading from dim (oldest) to
                              bright (newest) – solid filled.
      • Predicted positions : grey circles, slightly smaller – indicates the
                              BallTracker extended the path between detections.

    Ball marker
    -----------
      • Detected  : white outer ring + orange filled centre + "BALL" label.
      • Predicted : grey ring (no fill) + "BALL~" label – signals estimation.

    Label banner
    ------------
    A two-line banner occupies the bottom of the frame:
      • Line 1 (large, centred) – simplified phase name: "ATTACKING" / "DEFENDING"
        / "DELAYED SUPPORT" / "SPACING BREAKDOWN".  Background colour matches
        the tactical label (orange / blue / cyan / red).
      • Line 2 (small, right-aligned) – technical label, numeric index, and
        classifier confidence.

    Before the first sequence is classified a "CLASSIFYING…" placeholder is
    shown so the viewer knows the system is warming up.
    """
    out = frame.copy()
    h, w = out.shape[:2]

    # Resolve label-specific color
    box_color  = LABEL_COLORS.get(current_label, TRACK_COLOR)
    banner_bg  = LABEL_COLORS.get(current_label, LABEL_BG_COLOR)
    display_name = DISPLAY_NAMES.get(current_label, current_label) if current_label else ""

    # Net reference line to make side-of-net filtering explicit in output.
    net_y_px = int(round(h * (NET_Y_CM / COURT_H_CM)))
    cv2.line(out, (0, net_y_px), (w, net_y_px), (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(out, "NET", (10, max(20, net_y_px - 8)), FONT, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    # ── Ball trail (detected = orange fading, predicted = grey fading) ─────────
    trail_list = list(ball_trail)
    n_trail = len(trail_list)
    bc_b, bc_g, bc_r = BALL_COLOR   # BGR components of orange ball colour
    for i, (tx, ty, pred) in enumerate(trail_list):
        alpha = (i + 1) / max(n_trail, 1)       # 0 → oldest, 1 → newest
        radius = max(2, round(5 * alpha))
        if pred:
            # Predicted segment: grey, slightly smaller
            grey = int(160 * alpha)
            cv2.circle(out, (tx, ty), radius, (grey, grey, grey), -1, cv2.LINE_AA)
        else:
            # Detected segment: orange fading trail
            cv2.circle(out, (tx, ty), radius, (
                int(bc_b * alpha),
                int(bc_g * alpha),
                int(bc_r * alpha),
            ), -1, cv2.LINE_AA)

    # ── Player bounding boxes (colour reflects current tactical label) ─────────
    if tracks.tracker_id is not None and len(tracks.xyxy) > 0:
        for tid, (x1, y1, x2, y2) in zip(tracks.tracker_id, tracks.xyxy):
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(out, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(
                out, f"P {tid}", (x1, y1 - 6),
                FONT, 0.45, box_color, 1, cv2.LINE_AA,
            )

    # Large action-focus box highlighting the active team region.
    if current_label and tracks.tracker_id is not None and len(tracks.xyxy) > 0:
        x1 = int(np.min(tracks.xyxy[:, 0]))
        y1 = int(np.min(tracks.xyxy[:, 1]))
        x2 = int(np.max(tracks.xyxy[:, 2]))
        y2 = int(np.max(tracks.xyxy[:, 3]))

        pad_x = max(24, int(0.06 * max(1, x2 - x1)))
        pad_y = max(20, int(0.08 * max(1, y2 - y1)))
        ax1 = max(0, x1 - pad_x)
        ay1 = max(0, y1 - pad_y)
        ax2 = min(w - 1, x2 + pad_x)
        ay2 = min(h - 1, y2 + pad_y)

        overlay_focus = out.copy()
        cv2.rectangle(overlay_focus, (ax1, ay1), (ax2, ay2), box_color, -1)
        cv2.addWeighted(overlay_focus, 0.12, out, 0.88, 0, out)
        cv2.rectangle(out, (ax1, ay1), (ax2, ay2), box_color, 3)

        action_text = f"ACTION FOCUS: {display_name}"
        (tw, th), _ = cv2.getTextSize(action_text, FONT, 0.6, 2)
        tx = ax1 + 10
        ty = max(24, ay1 - 10)
        cv2.rectangle(out, (tx - 6, ty - th - 6), (tx + tw + 6, ty + 6), (0, 0, 0), -1)
        cv2.putText(out, action_text, (tx, ty), FONT, 0.6, box_color, 2, cv2.LINE_AA)

    # ── Ball marker ─────────────────────────────────────────────────────────────
    if ball_center is not None:
        cx, cy = int(ball_center[0]), int(ball_center[1])
        if ball_is_predicted:
            # Predicted: grey ring only – no fill, label with tilde
            cv2.circle(out, (cx, cy), 11, (160, 160, 160), 2, cv2.LINE_AA)
            cv2.putText(out, "BALL~", (cx + 13, cy + 5),
                        FONT, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
        else:
            # Detected: white ring + orange fill
            cv2.circle(out, (cx, cy), 11, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(out, (cx, cy), 8,  BALL_COLOR,      -1, cv2.LINE_AA)
            cv2.putText(out, "BALL", (cx + 13, cy + 5),
                        FONT, 0.5, BALL_COLOR, 1, cv2.LINE_AA)

    # ── Frame counter + timestamp ──────────────────────────────────────────────
    ts = frame_idx / fps
    cv2.putText(
        out, f"Frame {frame_idx:05d}  {ts:.2f}s",
        (10, 28), FONT, 0.65, TEXT_COLOR, 2, cv2.LINE_AA,
    )

    # ── Label banner at the bottom of the frame ────────────────────────────────
    big_scale, big_thick = 1.1, 2
    sm_scale,  sm_thick  = 0.55, 1
    (_, big_h), _ = cv2.getTextSize("Ag", FONT, big_scale, big_thick)
    (_, sm_h),  _ = cv2.getTextSize("Ag", FONT, sm_scale,  sm_thick)
    banner_h = big_h + sm_h + 24

    banner_y0 = h - banner_h - 4

    overlay = out.copy()
    cv2.rectangle(overlay, (0, banner_y0), (w, h), banner_bg, -1)
    cv2.addWeighted(overlay, 0.82, out, 0.18, 0, out)

    if current_label:
        numeric_idx = LABEL_TO_INDEX.get(current_label, 0)

        (dn_w, _), _ = cv2.getTextSize(display_name, FONT, big_scale, big_thick)
        big_x = max(10, (w - dn_w) // 2)
        big_y = banner_y0 + big_h + 8
        cv2.putText(out, display_name, (big_x, big_y),
                    FONT, big_scale, TEXT_COLOR, big_thick, cv2.LINE_AA)

        sub_text = f"[{numeric_idx}] {current_label}   conf: {current_conf:.2f}"
        (sub_w, _), _ = cv2.getTextSize(sub_text, FONT, sm_scale, sm_thick)
        sub_x = max(10, w - sub_w - 10)
        sub_y = banner_y0 + big_h + sm_h + 18
        cv2.putText(out, sub_text, (sub_x, sub_y),
                    FONT, sm_scale, TEXT_COLOR, sm_thick, cv2.LINE_AA)

    else:
        warmup = "CLASSIFYING..."
        (ww, _), _ = cv2.getTextSize(warmup, FONT, big_scale, big_thick)
        cv2.putText(out, warmup, (max(10, (w - ww) // 2), banner_y0 + big_h + 8),
                    FONT, big_scale, TEXT_COLOR, big_thick, cv2.LINE_AA)

    return out


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> None:
    """Execute the full analytics pipeline on a single match video."""

    # ── Dependency check ────────────────────────────────────────────────────
    if not HAS_ULTRALYTICS:
        print(
            "[ERROR] ultralytics is required.\n"
            "        Install with: pip install ultralytics>=8.0.0",
            file=sys.stderr,
        )
        sys.exit(1)
    if not HAS_SUPERVISION:
        print(
            "[ERROR] supervision is required.\n"
            "        Install with: pip install supervision>=0.18.0",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Device ──────────────────────────────────────────────────────────────
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    # ── Load models ─────────────────────────────────────────────────────────
    print(f"Loading YOLOv8 model : {args.yolo_model}")
    yolo = YOLO(args.yolo_model)

    print("Initialising ByteTrack tracker")
    tracker = sv.ByteTrack()

    model: MambaClassifier | None = None
    norm_mean: torch.Tensor | None = None
    norm_std: torch.Tensor | None = None

    if args.checkpoint:
        print(f"Loading Mamba checkpoint : {args.checkpoint}")
        try:
            model, norm_mean, norm_std = _load_mamba(args.checkpoint, device)
            print(f"Device: {device}")
        except FileNotFoundError:
            print(
                f"[WARNING] Checkpoint not found: {args.checkpoint}\n"
                "          Falling back to rule-based classification.",
                file=sys.stderr,
            )
    else:
        print("No Mamba checkpoint supplied – using rule-based classification.")

    # ── Homography ──────────────────────────────────────────────────────────
    H: np.ndarray | None = None
    if args.court_corners:
        H = _build_homography(args.court_corners)
        print("Court homography computed from supplied corner points.")
    else:
        print(
            f"No court corners supplied – pixel coords scaled to "
            f"{COURT_W_CM:.0f}×{COURT_H_CM:.0f} cm."
        )

    side_name = {
        "top": "TOP side of net",
        "bottom": "BOTTOM side of net",
        "all": "ALL players (no side filtering)",
    }[args.team_side]
    print(
        f"Team-side filter: {side_name} "
        f"(net margin: {args.net_margin_cm:.1f} cm)"
    )

    # ── Open video ───────────────────────────────────────────────────────────
    print(f"\nOpening video: {args.video}")
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {args.video}", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  {frame_w}×{frame_h}  {fps:.2f} FPS  ~{total_frames} frames")

    # ── Output video writer ──────────────────────────────────────────────────
    writer: cv2.VideoWriter | None = None
    if args.output_video:
        writer = cv2.VideoWriter(args.output_video, FOURCC, fps, (frame_w, frame_h))
        if not writer.isOpened():
            print(
                f"[WARNING] Cannot create output video: {args.output_video}. "
                "Skipping video output.",
                file=sys.stderr,
            )
            writer = None
        else:
            print(f"  Output video: {args.output_video}")

    # ── State variables ──────────────────────────────────────────────────────
    seq_buffer: list[np.ndarray] = []   # accumulates (14,) rows → 29 frames
    track_age: dict[int, int] = {}      # track_id → number of frames seen
    ball_tracker = BallTracker()        # constant-velocity ball tracker
    ball_trail: deque[tuple[int, int, bool]] = deque(maxlen=BALL_TRAIL_LEN)
    label_counts: dict[str, int] = defaultdict(int)
    predictions: list[dict] = []

    frame_idx = 0
    seq_idx = 0
    current_label = ""
    current_conf = 0.0

    print("\nProcessing frames…")

    # ── Main frame loop ──────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # ── STEP 1 – Frame Extraction (implicit in cap.read) ────────────────
        # ── STEP 2 – Player Detection (YOLOv8) ─────────────────────────────
        person_xyxy, person_conf, ball_center = _detect(
            frame, yolo, args.conf_threshold
        )

        person_xyxy, person_conf = _filter_players_by_team_side(
            person_xyxy,
            person_conf,
            frame_w,
            frame_h,
            H,
            args.team_side,
            args.net_margin_cm,
        )

        # ── STEP 3 – Player Tracking (ByteTrack) ────────────────────────────
        tracks = _track(person_xyxy, person_conf, tracker)

        # ── Ball tracking: bridge missed YOLO detections with predicted pos ──
        # BallTracker applies a constant-velocity kinematic model to fill
        # gaps (up to BALL_MAX_MISS_FRAMES) when the ball is not found by YOLO.
        tracked_ball, ball_is_predicted = ball_tracker.update(ball_center)

        # ── STEP 4 – Pose / Position Extraction ─────────────────────────────
        player_positions, ball_pos = _extract_positions(
            tracks, tracked_ball, frame_w, frame_h, H, track_age
        )

        # ── STEP 5 – Feature Engineering ────────────────────────────────────
        feature_row = _compute_frame_features(player_positions, ball_pos)
        seq_buffer.append(feature_row)

        # ── STEP 6 – Sequence Formation → STEP 7 Mamba → STEP 8 Classification
        if len(seq_buffer) == SEQ_LEN:
            seq_array = np.stack(seq_buffer)  # (29, 14)

            # Derived metrics (for output CSV)
            metrics = _sequence_metrics(seq_array)

            # Mamba classification or rule-based fallback
            if model is not None:
                label, numeric_idx, conf, probs = _classify_sequence(
                    seq_array, model, norm_mean, norm_std, device
                )
            else:
                label, numeric_idx, conf, probs = _classify_sequence_rule_based(seq_array)
            current_label = label
            current_conf = conf
            label_counts[label] += 1
            seq_idx += 1

            record = {
                "sequence": seq_idx,
                "frame_start": frame_idx - SEQ_LEN + 1,
                "frame_end": frame_idx,
                "label": label,
                "label_index": numeric_idx,
                "confidence": round(conf, 4),
                **metrics,
                **(
                    {f"p_{n}": round(probs[i].item(), 4) for i, n in enumerate(LABEL_NAMES)}
                    if probs is not None
                    else {}
                ),
            }
            predictions.append(record)

            display = DISPLAY_NAMES.get(label, label)
            print(
                f"  Seq {seq_idx:4d}  "
                f"frames {frame_idx - SEQ_LEN + 1:5d}–{frame_idx:5d}  "
                f"→  [{numeric_idx}] {display:<20s}  "
                f"conf={conf:.3f}  sync={metrics['sync_score']:+.3f}  "
                f"spacing={metrics['mean_spacing_cm']:.0f}cm"
            )

            seq_buffer.clear()

        # ── Annotate and write output frame ─────────────────────────────────
        # Update ball trail with this frame's tracked position (detected or predicted)
        if tracked_ball is not None:
            ball_trail.append((int(tracked_ball[0]), int(tracked_ball[1]), ball_is_predicted))

        if writer is not None:
            annotated = _annotate_frame(
                frame, tracks, tracked_ball, ball_is_predicted, ball_trail,
                current_label, current_conf, frame_idx, fps,
            )
            writer.write(annotated)

        if frame_idx % 300 == 0:
            pct = frame_idx / max(total_frames, 1) * 100
            print(f"  … {frame_idx}/{total_frames} frames ({pct:.0f}%)")

    cap.release()
    if writer is not None:
        writer.release()
        print(f"\nAnnotated video saved to '{args.output_video}'.")

    # ── Save predictions CSV ─────────────────────────────────────────────────
    if predictions and args.output_csv:
        pd.DataFrame(predictions).to_csv(args.output_csv, index=False)
        print(f"Predictions CSV saved to '{args.output_csv}'.")

    # ── STEP 9 – Final Tactical Report ──────────────────────────────────────
    print(f"\nTotal frames processed : {frame_idx}")
    print(f"Total sequences classified : {seq_idx}")

    if label_counts:
        print(generate_report(dict(label_counts)))
    else:
        print("\n[INFO] No complete sequences were classified "
              f"(need at least {SEQ_LEN} frames).")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

class _CornersAction(argparse.Action):
    """Parse four pixel corner coordinates passed as 'x0,y0 x1,y1 x2,y2 x3,y3'."""

    def __call__(self, parser, namespace, values, option_string=None):
        try:
            pairs = [tuple(float(v) for v in s.split(",")) for s in values]
            if len(pairs) != 4 or any(len(p) != 2 for p in pairs):
                raise ValueError
        except ValueError:
            parser.error(
                "--court-corners expects exactly four x,y pairs "
                "(e.g. 42,18 1238,18 1238,702 42,702)"
            )
        setattr(namespace, self.dest, pairs)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end volleyball analytics pipeline: "
            "YOLOv8 detection → ByteTrack tracking → Mamba classification "
            "→ tactical report."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("video", help="Path to the input match video (.mp4).")
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default=None,
        help=(
            "Path to a trained Mamba checkpoint (.pt) produced by train_mamba.py. "
            "If omitted or the file is not found, rule-based classification is used."
        ),
    )
    parser.add_argument(
        "--yolo-model",
        default="yolov8n.pt",
        help="YOLOv8 model weights (downloads automatically on first use).",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.35,
        help="YOLOv8 detection confidence threshold.",
    )
    parser.add_argument(
        "--court-corners",
        nargs=4,
        metavar=("TL", "TR", "BR", "BL"),
        action=_CornersAction,
        default=None,
        help=(
            "Four pixel coordinates (x,y) of the court boundary corners "
            "in TL→TR→BR→BL order, for perspective homography. "
            "Example: --court-corners 42,18 1238,18 1238,702 42,702"
        ),
    )
    parser.add_argument(
        "--team-side",
        choices=("bottom", "top", "all"),
        default="bottom",
        help=(
            "Which side of the net to treat as our team for tracking and "
            "classification features. Use 'bottom' for players before the net "
            "(near-camera side in common broadcast view)."
        ),
    )
    parser.add_argument(
        "--net-margin-cm",
        type=float,
        default=30.0,
        help=(
            "Safety margin around the net line when filtering by side. "
            "Higher values reduce cross-side leakage near the net."
        ),
    )
    parser.add_argument(
        "--output-video",
        default="",
        help="Path for the annotated output video (skipped if empty).",
    )
    parser.add_argument(
        "--output-csv",
        default="",
        help="Path to save per-sequence predictions as a CSV (skipped if empty).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_pipeline(_parse_args())
