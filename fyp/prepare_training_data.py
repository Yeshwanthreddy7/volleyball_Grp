"""
prepare_training_data.py - Build Mamba training CSVs from labeled clip folders.

Input layout (from extract_clips.py):
  dataset/
    Coordinated_Attack/
    Coordinated_Defense/
    Delayed_Support/
    Spacing_Breakdown/

Each clip is processed with YOLOv8 + ByteTrack, converted to 29-frame feature
sequences, and written as CSV files containing:
  frame_id, ball_x, ball_y, p1_x, p1_y, ..., p6_x, p6_y, target_label

Usage
-----
python prepare_training_data.py dataset --output-dir training_csv
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd

from pipeline import (
    COURT_H_CM,
    FEATURE_COLS,
    HAS_SUPERVISION,
    HAS_ULTRALYTICS,
    SEQ_LEN,
    BallTracker,
    _build_homography,
    _compute_frame_features,
    _detect,
    _extract_positions,
    _filter_players_by_team_side,
    _track,
)

if HAS_ULTRALYTICS:
    from ultralytics import YOLO
if HAS_SUPERVISION:
    import supervision as sv


FOLDER_TO_LABEL = {
    "Coordinated_Attack": "Coordinated Attack",
    "Coordinated_Defense": "Coordinated Defense",
    "Delayed_Support": "Delayed Support",
    "Spacing_Breakdown": "Spacing Breakdown",
}

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")


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


def _iter_clip_files(dataset_dir: str) -> list[tuple[str, str, str]]:
    """Return list of (clip_path, folder_name, target_label)."""
    clips: list[tuple[str, str, str]] = []
    for folder_name, label_name in FOLDER_TO_LABEL.items():
        class_dir = os.path.join(dataset_dir, folder_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):
            if not fname.lower().endswith(VIDEO_EXTS):
                continue
            clips.append((os.path.join(class_dir, fname), folder_name, label_name))
    return clips


def _extract_clip_sequence(
    clip_path: str,
    yolo: "YOLO",
    conf_threshold: float,
    team_side: str,
    net_margin_cm: float,
    H: np.ndarray | None,
) -> np.ndarray | None:
    """Extract fixed-length (SEQ_LEN, 14) feature sequence from one clip."""
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        return None

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tracker = sv.ByteTrack()
    track_age: dict[int, int] = {}
    ball_tracker = BallTracker()

    rows: list[np.ndarray] = []

    while len(rows) < SEQ_LEN:
        ret, frame = cap.read()
        if not ret:
            break

        person_xyxy, person_conf, ball_center = _detect(frame, yolo, conf_threshold)
        person_xyxy, person_conf = _filter_players_by_team_side(
            person_xyxy,
            person_conf,
            frame_w,
            frame_h,
            H,
            team_side,
            net_margin_cm,
        )

        tracks = _track(person_xyxy, person_conf, tracker)
        tracked_ball, _ = ball_tracker.update(ball_center)

        player_positions, ball_pos = _extract_positions(
            tracks,
            tracked_ball,
            frame_w,
            frame_h,
            H,
            track_age,
        )
        rows.append(_compute_frame_features(player_positions, ball_pos))

    cap.release()

    if not rows:
        return None

    seq = np.stack(rows).astype(np.float32)
    if len(seq) < SEQ_LEN:
        pad = np.zeros((SEQ_LEN - len(seq), seq.shape[1]), dtype=np.float32)
        seq = np.vstack([seq, pad])
    else:
        seq = seq[:SEQ_LEN]

    return seq


def run(args: argparse.Namespace) -> None:
    if not HAS_ULTRALYTICS:
        print("[ERROR] ultralytics is required. Install with: pip install ultralytics>=8.0.0", file=sys.stderr)
        sys.exit(1)
    if not HAS_SUPERVISION:
        print("[ERROR] supervision is required. Install with: pip install supervision>=0.18.0", file=sys.stderr)
        sys.exit(1)

    dataset_dir = os.path.abspath(args.dataset_dir)
    out_dir = os.path.abspath(args.output_dir)

    if not os.path.isdir(dataset_dir):
        print(f"[ERROR] dataset directory not found: {dataset_dir}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)

    if args.clean_output:
        removed = 0
        for fname in os.listdir(out_dir):
            if fname.lower().endswith(".csv"):
                os.remove(os.path.join(out_dir, fname))
                removed += 1
        print(f"Removed {removed} existing CSV files from '{out_dir}'.")

    H: np.ndarray | None = None
    if args.court_corners:
        H = _build_homography(args.court_corners)
        print("Using supplied court homography for coordinate mapping.")
    else:
        print("No court corners supplied. Using linear pixel-to-court scaling.")

    clips = _iter_clip_files(dataset_dir)
    if not clips:
        print(f"[ERROR] No video clips found under: {dataset_dir}")
        sys.exit(1)

    if args.max_clips > 0:
        clips = clips[: args.max_clips]

    print(f"Loading YOLO model: {args.yolo_model}")
    yolo = YOLO(args.yolo_model)

    print(f"Preparing {len(clips)} clips -> {out_dir}")
    success = 0
    skipped = 0

    for i, (clip_path, folder_name, label_name) in enumerate(clips, start=1):
        seq = _extract_clip_sequence(
            clip_path=clip_path,
            yolo=yolo,
            conf_threshold=args.conf_threshold,
            team_side=args.team_side,
            net_margin_cm=args.net_margin_cm,
            H=H,
        )
        if seq is None:
            print(f"[{i}/{len(clips)}] SKIP {os.path.basename(clip_path)}: could not extract frames")
            skipped += 1
            continue

        df = pd.DataFrame(seq, columns=FEATURE_COLS)
        df.insert(0, "frame_id", np.arange(1, SEQ_LEN + 1, dtype=int))
        df["target_label"] = label_name

        stem = os.path.splitext(os.path.basename(clip_path))[0]
        out_name = f"{folder_name}__{stem}.csv"
        out_path = os.path.join(out_dir, out_name)
        df.to_csv(out_path, index=False)

        print(f"[{i}/{len(clips)}] OK   {os.path.basename(clip_path)} -> {out_name}")
        success += 1

    print(f"\nDone. CSV prepared: {success}, skipped: {skipped}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Mamba training CSVs from labeled volleyball clips.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dataset_dir",
        help="Directory containing class folders of clips (dataset root).",
    )
    parser.add_argument(
        "--output-dir",
        default="training_csv",
        help="Directory where output CSV files will be written.",
    )
    parser.add_argument(
        "--yolo-model",
        default="yolov8n.pt",
        help="YOLOv8 model weights path.",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.35,
        help="YOLO detection confidence threshold.",
    )
    parser.add_argument(
        "--team-side",
        choices=("bottom", "top", "all"),
        default="bottom",
        help="Which side of the net to treat as our team.",
    )
    parser.add_argument(
        "--net-margin-cm",
        type=float,
        default=30.0,
        help="Net-margin exclusion zone for side filtering.",
    )
    parser.add_argument(
        "--court-corners",
        nargs=4,
        metavar=("TL", "TR", "BR", "BL"),
        action=_CornersAction,
        default=None,
        help="Optional court corners for homography mapping.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete existing CSV files from output directory before writing.",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=0,
        help="Limit number of clips to process (0 means all).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(_parse_args())
