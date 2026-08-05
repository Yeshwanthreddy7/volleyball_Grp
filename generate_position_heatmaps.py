"""
generate_position_heatmaps.py - Ball / court position heatmaps + base frames
for the two newly-added source videos: "videoplayback (3)" and
"videoplayback (4)".

Position data source
---------------------
Re-running full YOLO detection on the raw broadcast videos (1.3 GB / 552 MB)
is unnecessary: `training_csv/` already holds hundreds of 41-frame tracking
clips extracted from these exact videos (ball_x, ball_y, p1_x..p6_y in the
project's 1800x900 cm court-plane coordinate system - see fyp/court.py).
This script aggregates every clip whose filename references each source
video and turns the pooled (x, y) samples into density heatmaps.

Outputs (written to heatmaps/)
-------------------------------
  videoplayback_(3)_base_frame.png     - representative raw video frame
  videoplayback_(3)_ball_heatmap.png   - ball position density on the court
  videoplayback_(3)_court_heatmap.png  - all-player position density
  videoplayback_(4)_base_frame.png
  videoplayback_(4)_ball_heatmap.png
  videoplayback_(4)_court_heatmap.png
"""

from __future__ import annotations

import glob
import os

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(ROOT, "training_csv")
DATASET_DIR = os.path.join(ROOT, "dataset")
OUT_DIR = os.path.join(ROOT, "heatmaps")
os.makedirs(OUT_DIR, exist_ok=True)

VIDEOS = {
    "videoplayback (3)": os.path.join(DATASET_DIR, "videoplayback (3).mp4"),
    "videoplayback (4)": os.path.join(DATASET_DIR, "videoplayback (4).mp4"),
}

# Court plane constants (fyp/court.py - the live, tested calibration target).
COURT_W_CM = 1800.0
COURT_H_CM = 900.0
NET_Y_CM = COURT_H_CM / 2.0          # 450
ATTACK_LINE_OFFSET = 300.0           # 3 m attack line, each side of the net

# ---------------------------------------------------------------------------
# dataviz-skill palette: sequential single-hue ramps (light -> dark)
# Blue = default sequential hue (ball). Orange = next categorical slot,
# used as its own one-hue ramp for the second simultaneous sequential
# context (players/court occupancy).
# ---------------------------------------------------------------------------
BLUE_CMAP = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"]
)
ORANGE_CMAP = LinearSegmentedColormap.from_list(
    "seq_orange", ["#fce3d6", "#f3a880", "#eb6834", "#b84f26", "#7a3319"]
)

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "Arial"]


# ---------------------------------------------------------------------------
# 1. Aggregate tracking data for one source video across every clip/category
# ---------------------------------------------------------------------------
def load_positions(video_tag: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (ball_xy, player_xy) arrays pooled from every training_csv clip
    that was extracted from *video_tag* (e.g. 'videoplayback (3)')."""
    pattern = os.path.join(CSV_DIR, f"*{video_tag}*_clip_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No clip CSVs found for {video_tag!r} in {CSV_DIR}")

    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)

    # Ball: drop the (0,0) missing-detection sentinel, then clip obvious
    # homography-noise outliers (airborne ball frames) to a generous margin
    # around the nominal court plane.
    bx, by = df["ball_x"].to_numpy(), df["ball_y"].to_numpy()
    ball_mask = ~((bx == 0) & (by == 0))
    bx, by = bx[ball_mask], by[ball_mask]
    margin_x, margin_y = COURT_W_CM * 0.5, COURT_H_CM * 0.5
    bounds = (
        (bx >= -margin_x) & (bx <= COURT_W_CM + margin_x)
        & (by >= -margin_y) & (by <= COURT_H_CM + margin_y)
    )
    ball_xy = np.column_stack([bx[bounds], by[bounds]])

    # Players: pool p1..p6, drop the (0,0) missing/occluded sentinel.
    px = df.filter(regex=r"^p[1-6]_x$").to_numpy().ravel()
    py = df.filter(regex=r"^p[1-6]_y$").to_numpy().ravel()
    pmask = ~((px == 0) & (py == 0))
    player_xy = np.column_stack([px[pmask], py[pmask]])

    return ball_xy, player_xy


# ---------------------------------------------------------------------------
# 2. Heatmap rendering
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

    bins_x, bins_y = 72, 36
    hist, xedges, yedges = np.histogram2d(
        xy[:, 0], xy[:, 1],
        bins=[bins_x, bins_y],
        range=[[-40, COURT_W_CM + 40], [-40, COURT_H_CM + 40]],
    )
    density = gaussian_filter(hist.T, sigma=1.3)
    density = density / density.max() if density.max() > 0 else density
    density = np.ma.masked_less(density, 0.03)  # let empty court show through

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


# ---------------------------------------------------------------------------
# 3. Base frame extraction from the raw video
# ---------------------------------------------------------------------------
def extract_base_frame(video_path: str, out_path: str, frac: float = 0.12) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target = max(int(total * frac), 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ok, frame = cap.read()
    if not ok:
        # fall back to the very first frame if seeking failed
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read a frame from {video_path}")
    cv2.imwrite(out_path, frame)
    print(f"  wrote {out_path}  (frame {target}/{total})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    for tag, video_path in VIDEOS.items():
        safe = tag.replace(" ", "_")
        print(f"\n=== {tag} ===")

        ball_xy, player_xy = load_positions(tag)
        print(f"  ball samples: {len(ball_xy)}   player samples: {len(player_xy)}")

        render_heatmap(
            ball_xy, BLUE_CMAP,
            title=f"Ball Position Heatmap - {tag}",
            subtitle=f"{len(ball_xy)} tracked ball positions pooled across all labelled clips",
            out_path=os.path.join(OUT_DIR, f"{safe}_ball_heatmap.png"),
        )
        render_heatmap(
            player_xy, ORANGE_CMAP,
            title=f"Court Position Heatmap (All Players) - {tag}",
            subtitle=f"{len(player_xy)} tracked player positions (p1-p6) pooled across all labelled clips",
            out_path=os.path.join(OUT_DIR, f"{safe}_court_heatmap.png"),
        )

        if os.path.exists(video_path):
            extract_base_frame(video_path, os.path.join(OUT_DIR, f"{safe}_base_frame.png"))
        else:
            print(f"  [!] video file not found, skipping base frame: {video_path}")

    print("\nDone. All outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()
