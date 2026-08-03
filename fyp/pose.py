"""
pose.py – Individual biomechanical features from 17-point skeletal keypoints
(spec §1A).

What it measures, per tracked player
------------------------------------
* HIP-CENTROID VERTICAL VELOCITY  V_y  (m/s, upward positive)
    The mean of the two hip keypoints is the best single proxy for the body's
    centre of mass.  Its vertical velocity detects the explosive countermovement
    jump: an elite spike take-off leaves the ground at ~3.1–3.6 m/s (jump
    heights of 0.50–0.65 m, v = sqrt(2·g·h)), while ordinary court movement
    stays below ~1 m/s vertical.  A player is flagged AIRBORNE while the
    smoothed V_y exceeds `jump_vy_ms`.

* SHOULDER→WRIST ANGULAR VELOCITY  ω  (rad/s)
    The arm-swing angle (atan2 of the shoulder-to-wrist vector) differentiated
    over time.  The spike arm swing is the fastest reliably visible event in
    volleyball broadcast video; ω spiking above `impact_omega` localises the
    moment of ball strike to ±1 frame, which anchors the "Delayed Support"
    reaction-time analysis.

Vertical scale calibration
--------------------------
The floor homography maps the GROUND PLANE only – a jumping body leaves that
plane, so its height cannot come from the homography.  We convert pixels to
metres with a per-player scale: (assumed standing height / bounding-box height
in px).  Broadcast volleyball players average ~1.85 m minus posture crouch, so
`player_height_m` = 1.85 by default; the resulting V_y error is bounded by the
height-assumption error (<10 %), which is fine for jump DETECTION (a binary
event 3× above the noise floor).

Keypoint indexing = COCO-17:
  5 L-shoulder, 6 R-shoulder, 9 L-wrist, 10 R-wrist, 11 L-hip, 12 R-hip.
"""

from __future__ import annotations

import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

# COCO-17 keypoint ids
L_SHOULDER, R_SHOULDER = 5, 6
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12

KEYPOINT_CONF_MIN = 0.30

# Optional extension of the model feature vector (kept OUT of the default
# contract: the historical training CSVs carry no pose data, and injecting
# zeros at train time but real values at inference would break train/serve
# parity).  Enable only after re-extracting training data with pose.
POSE_FEATURE_COLS = [
    "pose_max_vy_up",      # max upward hip V_y among players (m/s)
    "pose_mean_abs_vy",    # mean |V_y| (m/s)
    "pose_max_arm_omega",  # max shoulder->wrist angular velocity (rad/s)
    "pose_n_airborne",     # players currently flagged airborne
]


@dataclass
class PoseFrameSummary:
    """Aggregated biomechanics for one frame (permutation-invariant)."""
    max_vy_up: float = 0.0
    mean_abs_vy: float = 0.0
    max_arm_omega: float = 0.0
    n_airborne: int = 0
    jump_ids: list[int] = field(default_factory=list)
    impact_ids: list[int] = field(default_factory=list)
    per_track: dict[int, dict] = field(default_factory=dict)

    def as_features(self) -> np.ndarray:
        return np.array(
            [self.max_vy_up, self.mean_abs_vy, self.max_arm_omega,
             float(self.n_airborne)],
            dtype=np.float32,
        )


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N,4)x(M,4) xyxy IoU matrix."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax1, ay1, ax2, ay2 = a[:, 0, None], a[:, 1, None], a[:, 2, None], a[:, 3, None]
    bx1, by1, bx2, by2 = b[None, :, 0], b[None, :, 1], b[None, :, 2], b[None, :, 3]
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / np.clip(area_a + area_b - inter, 1e-9, None)


def _wrap_angle(d: float) -> float:
    """Wrap an angle difference to (-pi, pi]."""
    while d > np.pi:
        d -= 2 * np.pi
    while d <= -np.pi:
        d += 2 * np.pi
    return d


class PoseEstimator:
    """
    17-keypoint pose extraction + per-track biomechanical state.

    Call `process(frame, tracked_xyxy, track_ids)` once per frame; it runs the
    pose model, matches skeletons to tracker ids by box IoU, updates per-track
    kinematic histories, and returns a PoseFrameSummary.

    The pose model is any ultralytics *-pose checkpoint ('yolo11n-pose.pt'
    default: 17-keypoint COCO head, real-time on CPU).  If the model cannot be
    loaded, the estimator disables itself and every summary is zeros – the
    pipeline keeps running (graceful degradation, spec "no exceptions").
    """

    def __init__(
        self,
        weights: str = "yolo11n-pose.pt",
        conf: float = 0.30,
        fps: float = 30.0,
        player_height_m: float = 1.85,
        jump_vy_ms: float = 1.8,
        impact_omega: float = 12.0,
        ema_alpha: float = 0.5,
        device: str | None = None,
    ) -> None:
        self._fps = fps
        self._h_m = player_height_m
        self._jump_vy = jump_vy_ms
        self._impact_omega = impact_omega
        self._alpha = ema_alpha
        self._conf = conf
        self.enabled = True
        self.events: list[dict] = []       # jump / impact event log
        self._frame_idx = 0

        # per-track state: hip_y px, bbox h px, arm angle rad, smoothed vy
        self._hist: dict[int, deque] = defaultdict(lambda: deque(maxlen=5))
        self._vy_ema: dict[int, float] = defaultdict(float)
        self._airborne: dict[int, bool] = defaultdict(bool)

        try:
            from ultralytics import YOLO
            self._model = YOLO(weights)
            if device:
                self._model.to(device)
        except Exception as exc:
            print(f"[WARNING] Pose model '{weights}' unavailable ({exc}); "
                  "biomechanical features disabled.", file=sys.stderr)
            self._model = None
            self.enabled = False

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._hist.clear()
        self._vy_ema.clear()
        self._airborne.clear()
        self.events.clear()
        self._frame_idx = 0

    # ------------------------------------------------------------------
    def process(
        self,
        frame: np.ndarray,
        tracked_xyxy: np.ndarray,
        track_ids: np.ndarray,
    ) -> PoseFrameSummary:
        self._frame_idx += 1
        if not self.enabled or len(tracked_xyxy) == 0:
            return PoseFrameSummary()

        try:
            res = self._model(frame, verbose=False, conf=self._conf)[0]
        except Exception as exc:  # pragma: no cover
            print(f"[WARNING] pose inference failed ({exc}); disabling pose.",
                  file=sys.stderr)
            self.enabled = False
            return PoseFrameSummary()

        if res.keypoints is None or res.boxes is None or len(res.boxes) == 0:
            return PoseFrameSummary()

        pose_xyxy = res.boxes.xyxy.cpu().numpy()
        kpts = res.keypoints.data.cpu().numpy()      # (M, 17, 3)

        # Greedy IoU match: tracker box  <->  pose box
        iou = _iou_matrix(np.asarray(tracked_xyxy, float), pose_xyxy)
        assign: dict[int, int] = {}
        flat = np.argsort(iou, axis=None)[::-1]
        used_t, used_p = set(), set()
        for f in flat:
            ti, pi = np.unravel_index(f, iou.shape)
            if iou[ti, pi] < 0.3 or ti in used_t or pi in used_p:
                continue
            assign[int(track_ids[ti])] = int(pi)
            used_t.add(ti); used_p.add(pi)

        summary = PoseFrameSummary()
        vys, omegas = [], []

        for tid, pi in assign.items():
            kp = kpts[pi]                            # (17, 3)
            box = pose_xyxy[pi]
            box_h = max(float(box[3] - box[1]), 1.0)
            m_per_px = self._h_m / box_h            # per-player vertical scale

            # hip centroid (needs both hips or one confident hip)
            hips = [kp[i] for i in (L_HIP, R_HIP) if kp[i, 2] >= KEYPOINT_CONF_MIN]
            hip_y = float(np.mean([h[1] for h in hips])) if hips else None

            # dominant arm angle: pick the (shoulder, wrist) pair with higher conf
            angle = None
            best_c = 0.0
            for s_i, w_i in ((L_SHOULDER, L_WRIST), (R_SHOULDER, R_WRIST)):
                c = min(kp[s_i, 2], kp[w_i, 2])
                if c >= KEYPOINT_CONF_MIN and c > best_c:
                    best_c = c
                    dx = kp[w_i, 0] - kp[s_i, 0]
                    dy = kp[w_i, 1] - kp[s_i, 1]
                    angle = float(np.arctan2(dy, dx))

            prev = self._hist[tid][-1] if self._hist[tid] else None
            vy = 0.0
            omega = 0.0
            if prev is not None:
                if hip_y is not None and prev["hip_y"] is not None:
                    # image y grows DOWNWARD -> upward velocity = -(dy)
                    vy_raw = -(hip_y - prev["hip_y"]) * m_per_px * self._fps
                    self._vy_ema[tid] = (
                        self._alpha * vy_raw
                        + (1 - self._alpha) * self._vy_ema[tid]
                    )
                    vy = self._vy_ema[tid]
                if angle is not None and prev["angle"] is not None:
                    omega = abs(_wrap_angle(angle - prev["angle"])) * self._fps

            self._hist[tid].append({"hip_y": hip_y, "angle": angle})

            airborne = vy > self._jump_vy
            if airborne and not self._airborne[tid]:
                summary.jump_ids.append(tid)
                self.events.append({
                    "frame": self._frame_idx, "track_id": tid,
                    "event": "jump_takeoff", "vy_ms": round(vy, 2),
                })
            self._airborne[tid] = airborne

            if omega > self._impact_omega:
                summary.impact_ids.append(tid)
                self.events.append({
                    "frame": self._frame_idx, "track_id": tid,
                    "event": "arm_swing_impact", "omega_rad_s": round(omega, 1),
                })

            summary.per_track[tid] = {
                "vy_ms": round(vy, 3),
                "arm_omega_rad_s": round(omega, 2),
                "airborne": airborne,
            }
            vys.append(vy)
            omegas.append(omega)

        if vys:
            summary.max_vy_up = round(max(max(vys), 0.0), 3)
            summary.mean_abs_vy = round(float(np.mean(np.abs(vys))), 3)
            summary.n_airborne = int(sum(self._airborne[t] for t in assign))
        if omegas:
            summary.max_arm_omega = round(max(omegas), 2)

        return summary


__all__ = ["PoseEstimator", "PoseFrameSummary", "POSE_FEATURE_COLS"]
