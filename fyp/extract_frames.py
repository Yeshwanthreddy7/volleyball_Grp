"""
extract_frames.py - extract frames from a match video for annotation.

Pulls frames at a fixed stride and, optionally, keeps only a DIVERSE subset
(frames that differ enough from the last kept frame) so you don't annotate
hundreds of near-identical frames - the redundancy that wastes labelling effort
and overfits the detector.

Usage
-----
  # every 30th frame (1 fps from 30 fps video) into ./frames
  python extract_frames.py "videoplayback (4).mp4" --output-dir frames --stride 30

  # ~1000 diverse frames
  python extract_frames.py "videoplayback (4).mp4" --output-dir frames \
      --stride 10 --diverse --diff-threshold 12 --max-frames 1000
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np


def _frame_signature(frame: np.ndarray) -> np.ndarray:
    """Small grayscale thumbnail used to measure frame-to-frame difference."""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (32, 32)).astype(np.float32)


def extract(video: str, out_dir: str, stride: int = 30, diverse: bool = False,
            diff_threshold: float = 12.0, max_frames: int = 0,
            start_frame: int = 0) -> int:
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"[ERROR] cannot open video: {video}")
    os.makedirs(out_dir, exist_ok=True)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    stem = os.path.splitext(os.path.basename(video))[0].replace(" ", "_")
    idx = start_frame
    kept = 0
    last_sig = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        if (idx - start_frame) % stride != 0:
            continue

        if diverse:
            sig = _frame_signature(frame)
            if last_sig is not None:
                mad = float(np.mean(np.abs(sig - last_sig)))
                if mad < diff_threshold:
                    continue          # too similar to the last kept frame
            last_sig = sig

        out_path = os.path.join(out_dir, f"{stem}_f{idx:06d}.jpg")
        cv2.imwrite(out_path, frame)
        kept += 1
        if kept % 100 == 0:
            print(f"  kept {kept} frames (at source frame {idx}/{total})")
        if max_frames and kept >= max_frames:
            break

    cap.release()
    print(f"Done. Extracted {kept} frames to '{out_dir}'.")
    return kept


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract (optionally diverse) frames for labelling.")
    p.add_argument("video", help="Input video path.")
    p.add_argument("--output-dir", default="frames", help="Where to write JPGs.")
    p.add_argument("--stride", type=int, default=30, help="Take every Nth frame.")
    p.add_argument("--diverse", action="store_true", help="Keep only frames that differ enough.")
    p.add_argument("--diff-threshold", type=float, default=12.0,
                   help="Min mean abs thumbnail difference to keep a frame (with --diverse).")
    p.add_argument("--max-frames", type=int, default=0, help="Stop after N kept frames (0=all).")
    p.add_argument("--start-frame", type=int, default=0, help="Begin at this source frame.")
    return p.parse_args()


if __name__ == "__main__":
    a = _parse_args()
    extract(a.video, a.output_dir, a.stride, a.diverse, a.diff_threshold,
            a.max_frames, a.start_frame)
