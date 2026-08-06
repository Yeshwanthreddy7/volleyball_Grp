"""
generate_video_position_heatmaps.py - Ball / court position heatmaps for
"videoplayback (3)" built DIRECTLY from the raw video in dataset/, via live
YOLO detection + court homography. Does NOT read training_csv at all - that
directory holds hand-labelled 41-frame tactic clips (a curated subset around
specific events), which is a different, narrower data source. This script
instead samples the full ~30 min match uniformly and detects the ball and
players on every sampled frame, so the resulting heatmaps reflect actual
positional coverage of the whole video, not just the labelled clip windows.

Pipeline (per sampled frame)
-----------------------------
  1. Court quad auto-detection (fyp.court.detect_court_quad) -> EMA-smoothed
     across frames to damp single-frame segmentation jitter (the camera is
     static broadcast, so the true quad barely moves).
  2. Homography to the near-half court plane (fyp.court.build_homography,
     dst=_HALF_DST) - auto-detected quads only ever capture the camera-near
     half (far half is compressed/net-occluded - see fyp/court.py docstring),
     the same near-side convention the rest of this project uses.
  3. YOLO detection:
       - players : yolo11n.pt (stock COCO 'person'), foot point = bbox
                    bottom-center.
       - ball    : fyp/volleyball_best.pt (fine-tuned 'ball' class), point =
                    bbox center. Domain fine-tune is materially better at
                    finding the volleyball than a stock detector (see
                    generate_position_heatmaps.py header for the same note).
  4. The ball is never gated on the in-court pixel polygon (see
     BALL_CM_BOUND / COURT_CM_BOUND below - a real rally ball sits above the
     net line in image space most of the time, outside that polygon by
     construction). Players ARE homographed past the calibrated near-half
     quad too - the auto-detected quad only ever covers the camera-near
     half (far half is compressed/net-occluded), so a strict polygon gate
     would silently drop the entire far team. Instead every player point is
     homographed and classified by which side of the net line (y=NET_Y_CM)
     it lands on, then kept if it clears the same generous COURT_CM_BOUND
     sanity check used for the ball. Far-side (opponent) positions are
     therefore an extrapolation past the calibrated quad - geometrically
     noisier than the near team's, since perspective compression is worse
     the farther a point sits from the calibrated half - but still a useful
     occupancy signal, not raw noise (see the homography extrapolation
     sanity-check in the dev notes).

Outputs (written to heatmaps/)
-------------------------------
  videoplayback_3_direct_detections.csv  - every kept (frame, t, x_cm, y_cm,
                                            conf, kind, team) sample, for audit.
  videoplayback_(3)_ball_heatmap.png     - ball position density (replaces
                                            the training_csv-sourced version)
  videoplayback_(3)_court_heatmap.png    - BOTH teams' player position
                                            density, overlaid: red = camera-
                                            near team (net behind the camera
                                            side), amber = far/opponent team.
"""

from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "fyp"))
from court import detect_court_quad, build_homography, order_corners, _HALF_DST  # noqa: E402

from ultralytics import YOLO  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
VIDEO_PATH = os.path.join(ROOT, "dataset", "videoplayback (3).mp4")
VIDEO_TAG = "videoplayback (3)"
OUT_DIR = os.path.join(ROOT, "heatmaps")
os.makedirs(OUT_DIR, exist_ok=True)

FRAME_STRIDE = 8          # ~6.25 fps sampling of a 50 fps source (~11.3k samples)
PERSON_MODEL_PATH = os.path.join(ROOT, "yolo11n.pt")
BALL_MODEL_PATH = os.path.join(ROOT, "fyp", "volleyball_best.pt")
PERSON_IMGSZ = 960
BALL_IMGSZ = 1280          # ball is tiny on 1080p footage; needs the larger res
PERSON_CONF = 0.35
BALL_CONF = 0.15
QUAD_EMA_ALPHA = 0.25       # smoothing weight for a NEW quad detection
COURT_MASK_MARGIN_PX = 20.0
# Post-homography sanity bound (cm), shared by ball AND players. Generous
# margin around the 1800x900 court: both the ball (airborne) and far-side
# players (past the calibrated near-half quad) get extrapolated by the same
# homography, which has no way to separate "far away on the floor" from
# "elevated but closer" - both push the projected point toward smaller y.
# This only rejects wildly-off extrapolations, not the far side itself.
COURT_CM_BOUND = (-300.0, 2100.0, -300.0, 1200.0)  # xmin, xmax, ymin, ymax

COURT_W_CM = 1800.0
COURT_H_CM = 900.0
NET_Y_CM = COURT_H_CM / 2.0
ATTACK_LINE_OFFSET = 300.0

DEVICE = 0  # GPU 0 (CUDA available - RTX 3050 confirmed at benchmark time)

# ---------------------------------------------------------------------------
# dataviz-skill palette - same ramps as generate_position_heatmaps.py so the
# two "families" of heatmap read as one visual system.
# ---------------------------------------------------------------------------
BLUE_CMAP = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"]
)
ORANGE_CMAP = LinearSegmentedColormap.from_list(
    "seq_orange", ["#fce3d6", "#f3a880", "#eb6834", "#b84f26", "#7a3319"]
)
# Red (near team) / amber (far team) - anchored on the palette's categorical
# red (#e34948) and yellow (#eda100) slots, same tint/shade construction as
# the orange ramp above. Validated as a 2-series light-surface pair via the
# dataviz skill's validate_palette.js: CVD dE 15.3, normal-vision dE 20.8
# (both PASS) - amber alone sits below 3:1 surface contrast, so it always
# ships with a labelled legend swatch, never color-alone.
RED_CMAP = LinearSegmentedColormap.from_list(
    "seq_red", ["#fbe3e3", "#ee9191", "#e34948", "#cb201f", "#8a1615"]
)
AMBER_CMAP = LinearSegmentedColormap.from_list(
    "seq_amber", ["#ffe1a1", "#ffc242", "#eda100", "#ac7500", "#614200"]
)
NEAR_TEAM_COLOR = "#e34948"
FAR_TEAM_COLOR = "#eda100"
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
BASELINE = "#c3c2b7"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "Arial"]


# ---------------------------------------------------------------------------
# Court quad tracking (EMA-smoothed auto-detection, static broadcast camera)
# ---------------------------------------------------------------------------
class SmoothedCourt:
    def __init__(self) -> None:
        self.quad: np.ndarray | None = None
        self.H: np.ndarray | None = None
        self.n_hits = 0
        self.n_misses = 0

    def update(self, frame: np.ndarray) -> None:
        raw = detect_court_quad(frame)
        if raw is None:
            self.n_misses += 1
            return
        self.n_hits += 1
        if self.quad is None:
            self.quad = raw
        else:
            self.quad = (1 - QUAD_EMA_ALPHA) * self.quad + QUAD_EMA_ALPHA * raw
        self.H = build_homography(self.quad, dst=_HALF_DST)

    def in_court(self, px: float, py: float) -> bool:
        if self.quad is None:
            return False
        return cv2.pointPolygonTest(
            self.quad.astype(np.float32), (float(px), float(py)), True
        ) >= -COURT_MASK_MARGIN_PX

    def to_court_cm(self, px: float, py: float) -> tuple[float, float]:
        pt = np.array([[[px, py]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self.H)
        return float(out[0, 0, 0]), float(out[0, 0, 1])


# ---------------------------------------------------------------------------
# Detection pass over the full video
# ---------------------------------------------------------------------------
def run_detection() -> pd.DataFrame:
    print(f"Loading models...\n  players: {PERSON_MODEL_PATH}\n  ball   : {BALL_MODEL_PATH}")
    person_model = YOLO(PERSON_MODEL_PATH)
    ball_model = YOLO(BALL_MODEL_PATH)
    ball_class_id = next(i for i, n in ball_model.names.items() if "ball" in str(n).lower())

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {VIDEO_PATH}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {VIDEO_PATH}\n  frames={total}  fps={fps:.1f}  "
          f"duration={total / fps / 60:.1f} min  stride={FRAME_STRIDE}")

    court = SmoothedCourt()
    rows: list[dict] = []
    frame_idx = 0
    t0 = time.time()
    n_samples = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % FRAME_STRIDE != 0:
            frame_idx += 1
            continue

        court.update(frame)
        if court.H is not None:
            # -- players --------------------------------------------------
            pr = person_model.predict(
                frame, imgsz=PERSON_IMGSZ, conf=PERSON_CONF, classes=[0],
                device=DEVICE, half=True, verbose=False,
            )[0]
            # -- players -------------------------------------------------
            # Homograph EVERY detection (no in-court pixel-polygon gate) -
            # the calibrated quad only covers the near half, so gating on it
            # would silently drop the entire far team. Classify each point
            # by which side of the net line (net_y_cm) it lands on, and keep
            # it if it clears the shared sanity bound.
            xmin, xmax, ymin, ymax = COURT_CM_BOUND
            for box in pr.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                fx, fy = (x1 + x2) / 2.0, y2
                cx, cy = court.to_court_cm(fx, fy)
                if xmin <= cx <= xmax and ymin <= cy <= ymax:
                    team = "near" if cy >= NET_Y_CM else "far"
                    rows.append({
                        "frame": frame_idx, "t_s": frame_idx / fps,
                        "kind": "player", "team": team, "x_cm": cx, "y_cm": cy,
                        "conf": float(box.conf[0]),
                    })

            # -- ball ------------------------------------------------------
            # Same treatment: the ball spends most of a rally above the net
            # line in image space, i.e. outside the near-half floor quad by
            # construction, so it is never gated on the pixel polygon.
            br = ball_model.predict(
                frame, imgsz=BALL_IMGSZ, conf=BALL_CONF, classes=[ball_class_id],
                device=DEVICE, half=True, verbose=False,
            )[0]
            if len(br.boxes):
                best = max(br.boxes, key=lambda b: float(b.conf[0]))
                x1, y1, x2, y2 = best.xyxy[0].tolist()
                bx, by = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                cx, cy = court.to_court_cm(bx, by)
                if xmin <= cx <= xmax and ymin <= cy <= ymax:
                    rows.append({
                        "frame": frame_idx, "t_s": frame_idx / fps,
                        "kind": "ball", "team": "", "x_cm": cx, "y_cm": cy,
                        "conf": float(best.conf[0]),
                    })

        n_samples += 1
        if n_samples % 250 == 0:
            elapsed = time.time() - t0
            rate = n_samples / elapsed
            remaining = (total // FRAME_STRIDE - n_samples) / max(rate, 1e-6)
            print(f"  sampled {n_samples} frames (video frame {frame_idx}/{total}, "
                  f"{frame_idx / total * 100:.1f}%) - {elapsed:.0f}s elapsed, "
                  f"~{remaining / 60:.1f} min remaining - "
                  f"quad hits/misses {court.n_hits}/{court.n_misses}",
                  flush=True)

        frame_idx += 1

    cap.release()
    elapsed = time.time() - t0
    print(f"\nDetection pass done: {n_samples} sampled frames in {elapsed / 60:.1f} min "
          f"({len(rows)} kept detections). Quad hits/misses: "
          f"{court.n_hits}/{court.n_misses}")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "videoplayback_3_direct_detections.csv")
    df.to_csv(csv_path, index=False)
    print(f"Wrote raw detections -> {csv_path}")
    return df


# ---------------------------------------------------------------------------
# Heatmap rendering (same visual system as generate_position_heatmaps.py)
# ---------------------------------------------------------------------------
def draw_court(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.add_patch(plt.Rectangle(
        (0, 0), COURT_W_CM, COURT_H_CM, fill=False,
        edgecolor=BASELINE, linewidth=1.5, zorder=5,
    ))
    ax.axhline(NET_Y_CM, color=INK_PRIMARY, linewidth=2.2, zorder=5)
    for y in (NET_Y_CM - ATTACK_LINE_OFFSET, NET_Y_CM + ATTACK_LINE_OFFSET):
        ax.axhline(y, color=INK_MUTED, linewidth=1.1, linestyle=(0, (6, 4)), zorder=5)
    ax.set_xlim(-40, COURT_W_CM + 40)
    ax.set_ylim(-40, COURT_H_CM + 40)
    ax.set_xlabel("Court X (cm)", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("Court Y (cm)", color=INK_SECONDARY, fontsize=10)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)


def render_heatmap(xy: np.ndarray, cmap, title: str, subtitle: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    draw_court(ax)

    if len(xy) > 0:
        bins_x, bins_y = 72, 36
        hist, xedges, yedges = np.histogram2d(
            xy[:, 0], xy[:, 1],
            bins=[bins_x, bins_y],
            range=[[-40, COURT_W_CM + 40], [-40, COURT_H_CM + 40]],
        )
        density = gaussian_filter(hist.T, sigma=1.3)
        density = density / density.max() if density.max() > 0 else density
        density = np.ma.masked_less(density, 0.03)

        im = ax.imshow(
            density, origin="lower",
            extent=[-40, COURT_W_CM + 40, -40, COURT_H_CM + 40],
            cmap=cmap, alpha=0.92, zorder=3, aspect="auto",
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label("Relative position density", color=INK_SECONDARY, fontsize=9)
        cbar.ax.tick_params(colors=INK_MUTED, labelsize=8)
        cbar.outline.set_visible(False)

    fig.subplots_adjust(top=0.84, bottom=0.11, left=0.07, right=0.91)
    fig.suptitle(title, x=0.045, y=0.97, ha="left", va="top",
                 color=INK_PRIMARY, fontsize=14, fontweight="bold")
    fig.text(0.045, 0.905, subtitle, ha="left", va="top",
              color=INK_SECONDARY, fontsize=9.5)

    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {out_path}")


def _density(xy: np.ndarray) -> np.ndarray | None:
    if len(xy) == 0:
        return None
    bins_x, bins_y = 72, 36
    hist, _, _ = np.histogram2d(
        xy[:, 0], xy[:, 1],
        bins=[bins_x, bins_y],
        range=[[-40, COURT_W_CM + 40], [-40, COURT_H_CM + 40]],
    )
    density = gaussian_filter(hist.T, sigma=1.3)
    density = density / density.max() if density.max() > 0 else density
    return np.ma.masked_less(density, 0.05)


def render_dual_heatmap(
    near_xy: np.ndarray, far_xy: np.ndarray,
    near_label: str, far_label: str,
    title: str, subtitle: str, out_path: str,
) -> None:
    """Two overlaid single-hue density layers (near=red, far=amber) on one
    court plot, each at reduced alpha so overlap blends rather than occludes.
    A labelled legend carries identity - required per the dataviz skill,
    since ambiguous overlap and the amber layer's sub-3:1 surface contrast
    both rule out color-alone identification."""
    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    draw_court(ax)

    extent = [-40, COURT_W_CM + 40, -40, COURT_H_CM + 40]
    for xy, cmap in ((far_xy, AMBER_CMAP), (near_xy, RED_CMAP)):
        density = _density(xy)
        if density is not None:
            ax.imshow(
                density, origin="lower", extent=extent,
                cmap=cmap, alpha=0.72, zorder=3, aspect="auto",
            )

    legend_handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", markersize=11,
                   markerfacecolor=NEAR_TEAM_COLOR, markeredgewidth=0,
                   label=f"{near_label}  (n={len(near_xy)})"),
        plt.Line2D([0], [0], marker="s", linestyle="", markersize=11,
                   markerfacecolor=FAR_TEAM_COLOR, markeredgewidth=0,
                   label=f"{far_label}  (n={len(far_xy)})"),
    ]
    legend = ax.legend(
        handles=legend_handles, loc="upper right", frameon=True,
        facecolor=SURFACE, edgecolor="none", fontsize=9.5,
        labelcolor=INK_PRIMARY, handletextpad=0.6, borderpad=0.6,
    )
    legend.get_frame().set_alpha(0.92)

    fig.subplots_adjust(top=0.84, bottom=0.11, left=0.07, right=0.96)
    fig.suptitle(title, x=0.045, y=0.97, ha="left", va="top",
                 color=INK_PRIMARY, fontsize=14, fontweight="bold")
    fig.text(0.045, 0.905, subtitle, ha="left", va="top",
              color=INK_SECONDARY, fontsize=9.5)

    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> None:
    df = run_detection()

    ball_xy = df[df["kind"] == "ball"][["x_cm", "y_cm"]].to_numpy()
    near_xy = df[(df["kind"] == "player") & (df["team"] == "near")][["x_cm", "y_cm"]].to_numpy()
    far_xy = df[(df["kind"] == "player") & (df["team"] == "far")][["x_cm", "y_cm"]].to_numpy()
    print(f"\nball samples: {len(ball_xy)}   "
          f"near-team samples: {len(near_xy)}   far-team samples: {len(far_xy)}")

    safe = VIDEO_TAG.replace(" ", "_")
    render_heatmap(
        ball_xy, BLUE_CMAP,
        title=f"Ball Position Heatmap - {VIDEO_TAG}",
        subtitle=(f"{len(ball_xy)} ball detections - direct YOLO detection on the raw "
                  f"video ({FRAME_STRIDE}-frame stride); floor-plane projection, "
                  f"ball height not modelled from monocular video"),
        out_path=os.path.join(OUT_DIR, f"{safe}_ball_heatmap.png"),
    )
    render_dual_heatmap(
        near_xy, far_xy,
        near_label="Near team (camera behind)", far_label="Far team (opponent)",
        title=f"Court Position Heatmap - {VIDEO_TAG}",
        subtitle=(f"{len(near_xy) + len(far_xy)} player detections, both teams - direct "
                  f"YOLO detection on the raw video ({FRAME_STRIDE}-frame stride); far-team "
                  f"positions are homography-extrapolated past the calibrated near-half "
                  f"court quad and are geometrically noisier"),
        out_path=os.path.join(OUT_DIR, f"{safe}_court_heatmap.png"),
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
