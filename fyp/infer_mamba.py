"""
infer_mamba.py – Inference script: classify volleyball clips with a trained Mamba model.

Reads a directory of CSV files (raw, unlabelled – frames 1–29 only, or
full 41-frame clips as output by process_video.py's companion pipeline),
classifies each clip with the trained MambaClassifier, and either:

  • prints a summary table to stdout, or
  • writes a labelled copy of each CSV (appending a `target_label` column)

Behaviour mirrors label_clips.py so both pipelines can be compared
side-by-side.

Usage
-----
    python infer_mamba.py <checkpoint> <csv_directory> [options]

Example
-------
    python infer_mamba.py mamba_checkpoint.pt ./raw_clips --write
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from mamba_model import FEATURE_COLS, INPUT_DIM, LABEL_NAMES, MambaClassifier
from label_clips import LABEL_TO_INDEX, generate_report
from features import build_model_sequence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_checkpoint(path: str, device: torch.device) -> tuple[MambaClassifier, torch.Tensor, torch.Tensor]:
    """Load a checkpoint and return (model, norm_mean, norm_std)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt.get("feature_version") != "perm_invariant_v1":
        print(
            "[WARNING] Checkpoint predates the permutation-invariant feature "
            "pipeline; predictions will be INVALID. Retrain with train_mamba.py.",
            file=sys.stderr,
        )
    saved_args = ckpt.get("args", {})

    model = MambaClassifier(
        input_dim=saved_args.get("input_dim", INPUT_DIM),
        d_model=saved_args.get("d_model", 64),
        n_layers=saved_args.get("n_layers", 4),
        d_state=saved_args.get("d_state", 16),
        d_conv=saved_args.get("d_conv", 4),
        num_classes=len(LABEL_NAMES),
        dropout=0.0,  # no dropout at inference time
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    mean = ckpt["norm_mean"].to(device)
    std = ckpt["norm_std"].to(device)
    return model, mean, std


def _prepare_sequence(df: pd.DataFrame) -> torch.Tensor | None:
    """
    Extract the training window (up to 29 frames) from a dataframe.
    Returns a (29, 14) float32 tensor, or None if columns are missing.
    """
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        return None

    # Use only frames 1–29 if frame_id column exists
    if "frame_id" in df.columns:
        df = df[df["frame_id"] <= 29]

    raw = df[FEATURE_COLS].values.astype(np.float32)
    if len(raw) == 0:
        return None

    # Build the corrected, identity-aware, permutation-invariant model input –
    # identical to what train_mamba.py and pipeline.py feed the network.
    seq = build_model_sequence(raw, target_len=29)
    return torch.from_numpy(seq)  # (29, 14)


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def run_inference(args: argparse.Namespace) -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")

    try:
        model, mean, std = _load_checkpoint(args.checkpoint, device)
    except FileNotFoundError:
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)

    csv_files = [f for f in os.listdir(args.csv_dir) if f.lower().endswith(".csv")]
    if not csv_files:
        print(f"No CSV files found in '{args.csv_dir}'.")
        return

    results: list[dict] = []

    for fname in sorted(csv_files):
        fpath = os.path.join(args.csv_dir, fname)
        try:
            df = pd.read_csv(fpath)
        except (pd.errors.ParserError, OSError, ValueError) as exc:
            print(f"[SKIP] {fname}: {exc}")
            continue

        seq = _prepare_sequence(df)
        if seq is None:
            print(f"[SKIP] {fname}: missing required feature columns.")
            continue

        # Normalise with training statistics
        seq = (seq.to(device) - mean) / std  # (29, 14)

        with torch.no_grad():
            logits = model(seq.unsqueeze(0))      # (1, 4)
            probs = F.softmax(logits, dim=-1)[0]  # (4,)

        pred_idx = probs.argmax().item()
        pred_label = LABEL_NAMES[pred_idx]
        confidence = probs[pred_idx].item()
        numeric_idx = LABEL_TO_INDEX.get(pred_label, 0)

        results.append({
            "file": fname,
            "prediction": pred_label,
            "label_index": numeric_idx,
            "confidence": confidence,
            **{f"p_{n}": probs[i].item() for i, n in enumerate(LABEL_NAMES)},
        })

        print(
            f"[OK] {fname:40s}  →  [{numeric_idx}] {pred_label:25s}  (conf: {confidence:.3f})"
        )

        if args.write:
            # Append label to the training-window rows and overwrite
            if "frame_id" in df.columns:
                out_df = df[df["frame_id"] <= 29].copy()
            else:
                out_df = df.copy()
            out_df = out_df.reset_index(drop=True)
            out_df["target_label"] = pred_label
            out_df.to_csv(fpath, index=False)

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\nClassified {len(results)} clips.")
    if results:
        label_counts = {}
        for r in results:
            label_counts[r["prediction"]] = label_counts.get(r["prediction"], 0) + 1
        print("\nLabel distribution:")
        for label, cnt in sorted(label_counts.items()):
            numeric_idx = LABEL_TO_INDEX.get(label, 0)
            print(f"  [{numeric_idx}] {label:25s} : {cnt}")
        print(generate_report(label_counts))

    if args.output_csv:
        pd.DataFrame(results).to_csv(args.output_csv, index=False)
        print(f"\nResults written to '{args.output_csv}'.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify volleyball clips using a trained Mamba model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "checkpoint",
        help="Path to the trained model checkpoint (.pt file).",
    )
    parser.add_argument(
        "csv_dir",
        help="Directory of CSV clip files to classify.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Overwrite each CSV with the predicted target_label column appended.",
    )
    parser.add_argument(
        "--output_csv",
        default="",
        help="Optional path to save a summary CSV of all predictions.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_inference(_parse_args())
