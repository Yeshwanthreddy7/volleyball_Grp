"""
make_final_annotated_compilation.py - Cut out just the labels.csv event
windows from an already-annotated video and stitch them back-to-back into
one compilation.

Why this doesn't re-run detection
----------------------------------
ANNOTATED_VIDEO3.mp4 (built by fyp/pipeline.py + build_ground_truth_video.py)
already covers the first 10 minutes of videoplayback (3).mp4 with correct
player boxes/ids and the labels.csv tactic banners burned in. Every one of
labels.csv's 108 events for this video falls inside that window (last event
at 00:09:36), so this script only needs to CUT and CONCATENATE - it reuses
the expensive YOLO detection pass instead of repeating it.

Each event's window uses the exact same convention as
build_ground_truth_video.py's build_windows():
    Coordinated Defense : [anchor_time - 0.5s, anchor_time + 0.5s]
    everything else      : [anchor_time,        anchor_time + 1.0s]

Usage
-----
    python make_final_annotated_compilation.py \
        --video "HEATMAPS_OUTPUT/ANNOTATED_VIDEO3.mp4" \
        --labels labels.csv \
        --video-file "videoplayback (3).mp4" \
        --output "HEATMAPS_OUTPUT/FINAL_ANNOTATED.mp4"
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_ground_truth_video import build_windows, CLASS_STYLE  # noqa: E402


def _resolve_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="Already-annotated source video.")
    ap.add_argument("--labels", default="labels.csv")
    ap.add_argument("--video-file", required=True,
                     help="Value to match against labels.csv's video_file column.")
    ap.add_argument("--output", required=True)
    ap.add_argument("--pad-s", type=float, default=0.0,
                     help="Extra seconds of context padded onto each side of every "
                          "event window (default 0 = exact labels.csv window).")
    args = ap.parse_args()

    windows = build_windows(args.labels, args.video_file)
    flat: list[tuple[float, float, str]] = []
    for tactic, spans in windows.items():
        label = CLASS_STYLE[tactic][0]
        for start, end in spans:
            flat.append((max(0.0, start - args.pad_s), end + args.pad_s, label))
    flat.sort(key=lambda t: t[0])

    if not flat:
        raise SystemExit(f"[ERROR] No events found for video_file={args.video_file!r}")

    print(f"Video file    : {args.video_file}")
    print(f"Events found  : {len(flat)}")
    total_s = sum(e - s for s, e, _ in flat)
    print(f"Total runtime : {total_s:.1f}s ({total_s / 60:.1f} min) of compiled clips")

    ffmpeg = _resolve_ffmpeg()
    tmpdir = tempfile.mkdtemp(prefix="final_annotated_")
    segment_paths: list[str] = []

    try:
        for i, (start, end, label) in enumerate(flat, 1):
            dur = end - start
            seg_path = os.path.join(tmpdir, f"seg_{i:03d}.mp4")
            cmd = [
                ffmpeg, "-y", "-ss", f"{start:.3f}", "-i", args.video,
                "-t", f"{dur:.3f}",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-c:a", "aac",
                seg_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0 or not os.path.exists(seg_path):
                print(f"[ERROR] segment {i} ({label} @ {start:.2f}s-{end:.2f}s) failed:\n"
                      f"{result.stderr[-2000:]}", file=sys.stderr)
                raise SystemExit(1)
            segment_paths.append(seg_path)
            print(f"  [{i:3d}/{len(flat)}] {label:<22s} {start:7.2f}s - {end:7.2f}s  ({dur:.2f}s)  -> {seg_path}")

        concat_list = os.path.join(tmpdir, "concat_list.txt")
        with open(concat_list, "w", encoding="utf-8") as fh:
            for p in segment_paths:
                # ffmpeg concat demuxer wants forward slashes / escaped paths
                fh.write(f"file '{p.replace(chr(92), '/')}'\n")

        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        cmd = [
            ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", concat_list,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac",
            "-movflags", "+faststart",
            args.output,
        ]
        print(f"\nConcatenating {len(segment_paths)} clips -> {args.output} ...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[ERROR] concat failed:\n{result.stderr[-2000:]}", file=sys.stderr)
            raise SystemExit(1)

        print(f"\nDone -> {args.output}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
