"""
build_ground_truth_video.py - Burn the KNOWN tactical-event timestamps from
labels.csv onto the full raw match video, with zero model inference.

Why this exists
----------------
`labels.csv` is the hand-labelled ground truth every training_csv clip was
cut from (fyp/extract_clips.py / prepare_training_data.py). Each row is
(video_file, anchor_time, tactic_class). The clip windows used to build
training_csv are exactly:
    Coordinated Defense : [anchor_time - 0.5s, anchor_time + 0.5s]
    everything else      : [anchor_time,        anchor_time + 1.0s]
This script reproduces that same window logic and overlays a label banner
on the ORIGINAL video at those exact timestamps - one fixed on-screen row
per tactic class, so simultaneous events never collide. Because nothing is
detected/tracked/classified live, there is no bounding-box / team-split bug
to inherit - this is a straight, deterministic dub of the dataset labels
onto the footage, useful to (a) sanity-check the labels themselves against
the real video and (b) as a reference video for comparing what the live
pipeline.py model predicts on the same footage.

Usage
-----
    python build_ground_truth_video.py \
        --video "dataset/videoplayback (3).mp4" \
        --labels labels.csv \
        --output heatmaps/videoplayback_3_ground_truth.mp4

`--video-file` (the value matched against labels.csv's video_file column)
defaults to the basename of --video.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

import pandas as pd

CANONICAL_CLASSES = {
    "coordinated_attack": "Coordinated_Attack",
    "coordinated_defense": "Coordinated_Defense",
    "coordinated_defence": "Coordinated_Defense",
    "delayed_support": "Delayed_Support",
    "spacing_breakdown": "Spacing_Breakdown",
}

# label text, banner color (0xRRGGBB), fixed vertical row (px from top)
CLASS_STYLE = {
    "Coordinated_Attack":  ("COORDINATED ATTACK",  "0xEB6834", 40),
    "Coordinated_Defense": ("COORDINATED DEFENSE", "0x2A78D6", 96),
    "Delayed_Support":     ("DELAYED SUPPORT",     "0xEDA100", 152),
    "Spacing_Breakdown":   ("SPACING BREAKDOWN",   "0xE34948", 208),
}


def _normalize_tactic(raw: str) -> str:
    key = raw.strip().lower().replace(" ", "_")
    return CANONICAL_CLASSES.get(key, raw.strip())


def _parse_time_to_seconds(hms: str) -> float:
    hh, mm, ss = hms.strip().split(":")
    return int(hh) * 3600 + int(mm) * 60 + float(ss)


def _resolve_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        print(
            "[ERROR] ffmpeg not found on PATH and imageio-ffmpeg isn't installed.\n"
            "        Install it with:  python -m pip install imageio-ffmpeg",
            file=sys.stderr,
        )
        sys.exit(1)


def _escape_ffmpeg_path(path: str) -> str:
    """Escape a filesystem path for use inside an ffmpeg filtergraph string
    (drive-letter colon and backslashes both need escaping there)."""
    path = path.replace("\\", "/")
    path = path.replace(":", "\\:")
    return path


def _find_font() -> str | None:
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeuib.ttf",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def build_windows(labels_path: str, video_file: str) -> dict[str, list[tuple[float, float]]]:
    df = pd.read_csv(labels_path)
    df = df[df["video_file"].str.strip() == video_file]
    if df.empty:
        raise SystemExit(
            f"[ERROR] No rows in {labels_path!r} for video_file={video_file!r}. "
            f"Available: {sorted(pd.read_csv(labels_path)['video_file'].unique())}"
        )

    windows: dict[str, list[tuple[float, float]]] = {k: [] for k in CLASS_STYLE}
    for _, row in df.iterrows():
        tactic = _normalize_tactic(str(row["tactic_class"]))
        if tactic not in CLASS_STYLE:
            continue
        anchor = _parse_time_to_seconds(str(row["anchor_time"]))
        start = anchor - 0.5 if tactic == "Coordinated_Defense" else anchor
        end = start + 1.0
        windows[tactic].append((start, end))

    for k in windows:
        windows[k].sort()
    return windows


def build_filtergraph(windows: dict[str, list[tuple[float, float]]], fontfile: str | None) -> str:
    filters = []
    fontfile_opt = ""
    if fontfile:
        fontfile_opt = f"fontfile='{_escape_ffmpeg_path(fontfile)}':"

    for tactic, (label, color, y) in CLASS_STYLE.items():
        events = windows.get(tactic, [])
        if not events:
            continue
        enable_expr = "+".join(f"between(t\\,{s:.3f}\\,{e:.3f})" for s, e in events)
        filters.append(
            f"drawtext={fontfile_opt}text='{label}':fontcolor=white:fontsize=30:"
            f"box=1:boxcolor={color}@0.82:boxborderw=14:x=40:y={y}:"
            f"enable='{enable_expr}'"
        )

    # running timecode, top-right - always visible
    filters.append(
        f"drawtext={fontfile_opt}text='%{{pts\\:hms}}':fontcolor=white:fontsize=22:"
        f"box=1:boxcolor=black@0.55:boxborderw=8:x=w-tw-30:y=20"
    )
    return ",".join(filters)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="Path to the raw match video.")
    ap.add_argument("--labels", default="labels.csv", help="Path to labels.csv (default: labels.csv).")
    ap.add_argument("--video-file", default=None,
                     help="Value to match against labels.csv's video_file column "
                          "(default: basename of --video).")
    ap.add_argument("--output", required=True, help="Path for the annotated output video.")
    ap.add_argument("--crf", type=int, default=20, help="libx264 quality (lower = better/larger, default 20).")
    ap.add_argument("--preset", default="veryfast", help="libx264 speed preset (default veryfast).")
    args = ap.parse_args()

    video_file = args.video_file or os.path.basename(args.video)
    windows = build_windows(args.labels, video_file)

    total_events = sum(len(v) for v in windows.values())
    print(f"Video file    : {video_file}")
    for tactic, (label, _, _) in CLASS_STYLE.items():
        print(f"  {label:<22}: {len(windows[tactic])} events")
    print(f"  TOTAL               : {total_events} events")
    if total_events == 0:
        raise SystemExit("[ERROR] No events matched - nothing to annotate.")

    fontfile = _find_font()
    if not fontfile:
        print("[WARN] No system font found (arialbd.ttf/arial.ttf/segoeuib.ttf); "
              "letting ffmpeg use its default. If drawtext errors out, pass a "
              "font path in the script's _find_font() candidates list.")

    vf = build_filtergraph(windows, fontfile)
    ffmpeg_exe = _resolve_ffmpeg()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    cmd = [
        ffmpeg_exe, "-y", "-i", args.video,
        "-vf", vf,
        "-c:v", "libx264", "-preset", args.preset, "-crf", str(args.crf),
        "-c:a", "copy",
        "-movflags", "+faststart",
        args.output,
    ]
    print("\nRunning ffmpeg (this re-encodes the full video - it will take a while)...")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(f"[ERROR] ffmpeg exited with code {result.returncode}")
    print(f"\nDone -> {args.output}")


if __name__ == "__main__":
    main()
