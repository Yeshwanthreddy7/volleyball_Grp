"""
diagnose_data.py – quantify dataset health and the correctness loopholes.

Run:  python fyp/diagnose_data.py training_csv

Reports, across every clip CSV:
  • class balance
  • frames per clip
  • ball-missing fraction (the (0,0)/NaN sentinel rate)
  • coordinate ranges (exposes train/serve scale or convention mismatch)
  • IDENTITY-SWAP metric: median frame-to-frame displacement of slot p1
        before vs after features.recover_identity().  A large "before" value
        (hundreds of cm/frame) is physically impossible for a real player at
        30 FPS and is the fingerprint of slot-swapping.
"""
from __future__ import annotations

import collections
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import RAW_COLS, raw_frame_to_arrays, recover_identity  # noqa: E402


def main(directory: str) -> None:
    files = sorted(glob.glob(os.path.join(directory, "*.csv")))
    if not files:
        print(f"No CSVs in '{directory}'.")
        return

    dist = collections.Counter()
    nrows = collections.Counter()
    ball_missing, neg_y = [], 0
    ranges = {c: [] for c in ["ball_x", "ball_y", "p1_x", "p1_y"]}
    jump_before, jump_after = [], []

    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {os.path.basename(f)}: {e}")
            continue
        if "target_label" in df:
            dist[df["target_label"].iloc[0]] += 1
        nrows[len(df)] += 1
        if {"ball_x", "ball_y"} <= set(df.columns):
            ball_missing.append(float(((df.ball_x == 0) & (df.ball_y == 0)).mean()))
            if (df.ball_y < 0).any():
                neg_y += 1
        for c in ranges:
            if c in df:
                ranges[c] += df[c].tolist()

        if set(RAW_COLS) <= set(df.columns):
            raw = df[RAW_COLS].to_numpy(float)
            players, _ = raw_frame_to_arrays(raw)
            p1 = players[:, 0, :]
            jb = np.linalg.norm(np.diff(p1, axis=0), axis=1)
            jump_before.append(np.nanmedian(jb) if np.isfinite(jb).any() else np.nan)
            fixed = recover_identity(players)[:, 0, :]
            ja = np.linalg.norm(np.diff(fixed, axis=0), axis=1)
            jump_after.append(np.nanmedian(ja) if np.isfinite(ja).any() else np.nan)

    print("=" * 60)
    print(f"DATA HEALTH REPORT  —  {len(files)} clips in '{directory}'")
    print("=" * 60)
    print("Class balance        :", dict(dist))
    print("Frames per clip       :", dict(nrows))
    if ball_missing:
        print(f"Ball-missing fraction : mean {np.mean(ball_missing):.0%} per clip")
        print(f"Clips with ball_y < 0 : {neg_y}  (impossible under 0..900 court → "
              f"coordinate-convention mismatch with pipeline)")
    print("Coordinate ranges     :")
    for c, v in ranges.items():
        v = np.asarray(v, float)
        print(f"   {c:7s}: min {v.min():8.1f}  max {v.max():8.1f}  mean {v.mean():8.1f}")
    if jump_before:
        print("-" * 60)
        print("IDENTITY-SWAP metric (median cm slot-p1 moves per frame):")
        print(f"   BEFORE recover_identity : {np.nanmedian(jump_before):7.1f} cm/frame")
        print(f"   AFTER  recover_identity : {np.nanmedian(jump_after):7.1f} cm/frame")
        print("   (a real player at 30 FPS moves ~5–40 cm/frame; a large BEFORE "
              "value confirms slot-swapping corrupts raw per-slot velocity)")
    print("=" * 60)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "training_csv")
