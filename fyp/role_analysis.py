"""
role_analysis.py - individual skill -> team tactic mapping on real clip CSVs.

Reads every training CSV, repairs identity EXACTLY like the model does
(features.raw_frame_to_arrays -> recover_identity -> interpolate_gaps), infers
behavioural roles per slot (roles.py), and aggregates per-role skill metrics
per tactical class, next to their literature anchors.

Pure NumPy/pandas - no GPU, no torch. Usage:
    python fyp/role_analysis.py training_csv role_analysis.csv
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features import (  # noqa: E402
    INTERP_MAX_GAP, RAW_COLS, interpolate_gaps, raw_frame_to_arrays,
    recover_identity,
)
from roles import ROLE_LITERATURE, ROLES, infer_roles, role_skill_summary  # noqa: E402


def analyze_clip_csv(path: str, fps: float = 30.0) -> dict | None:
    df = pd.read_csv(path)
    cols = [c for c in RAW_COLS if c in df.columns]
    if len(cols) != len(RAW_COLS):
        return None
    raw = df[RAW_COLS].to_numpy(dtype=float)
    players, _ball = raw_frame_to_arrays(raw)
    tracked = recover_identity(players)
    tracked = interpolate_gaps(tracked, max_gap=INTERP_MAX_GAP)
    rec = infer_roles(tracked, fps=fps, team_side="bottom")
    agg = role_skill_summary(rec)
    agg["clip"] = os.path.basename(path)
    agg["target_label"] = (df["target_label"].iloc[0]
                           if "target_label" in df.columns else "?")
    return agg


def main(csv_dir: str, out_csv: str | None = None) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    rows = []
    for f in files:
        try:
            r = analyze_clip_csv(f)
        except Exception as exc:            # keep going; report at the end
            print("  [warn] {}: {}".format(os.path.basename(f), exc))
            r = None
        if r is not None:
            rows.append(r)
    if not rows:
        print("No usable CSVs under", csv_dir)
        return pd.DataFrame()
    df = pd.DataFrame(rows)

    print("\n=== Individual role -> team tactic mapping "
          "({} clips) ===".format(len(df)))
    per_class = df.groupby("target_label").agg(
        clips=("clip", "count"),
        attackers_per_clip=("n_attacker", "mean"),
        attacker_approach_cms=("attacker_approach_cms", "mean"),
        liberos_per_clip=("n_libero", "mean"),
        libero_lateral_cms=("libero_lateral_cms", "mean"),
        setter_displacement_cm=("setter_displacement_cm", "mean"),
        unknown_per_clip=("n_unknown", "mean"),
    ).round(1)
    print(per_class.to_string())

    print("\n=== Literature anchors (see Technical_Review_and_QA.md refs) ===")
    for key, (lo, hi, note) in ROLE_LITERATURE.items():
        print("  {:<22} {:>6.0f}-{:<6.0f} {}".format(key, lo, hi, note))

    measured = df["attacker_approach_cms"].dropna()
    if len(measured):
        in_band = ((measured >= 200) & (measured <= 400)).mean() * 100
        print("\nAttacker approach speed: mean {:.0f} cm/s over {} clips with a "
              "detected attacker;\n{:.0f}% fall in the physically plausible "
              "200-400 cm/s band around the elite 260-300 anchor."
              .format(measured.mean(), len(measured), in_band))

    if out_csv:
        df.to_csv(out_csv, index=False)
        print("\nPer-clip role analysis written to", out_csv)
    return df


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "training_csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "role_analysis.csv"
    main(d, out)
