"""
prepare_training_data.py - Build Mamba training CSVs from labeled clip folders.

Input layout (from extract_clips.py):
  dataset/
    Coordinated_Attack/
    Coordinated_Defense/
    Delayed_Support/
    Spacing_Breakdown/

Each clip is processed with the abstract detector + tracker adapter stack,
converted to 29-frame feature sequences, and written as CSV files containing:
  frame_id, <FEATURE_COLS...>, target_label

Usage
-----
python prepare_training_data.py dataset --output-dir training_csv
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

# Benign, known warning from ultralytics tracker internals: a coasting
# (occluded) track can inflate its Kalman covariance until square() overflows.
# Downstream guards (SlotManager finite check, features._sanitize envelope,
# physio speed caps) already discard non-finite values, so silence the noise.
warnings.filterwarnings("ignore", message="overflow encountered in square")

import cv2
import numpy as np
import pandas as pd

from court import CourtCalibrator
from interfaces import DEFAULT_IMGSZ, create_detector
from identity import SlotManager
from tracking import create_tracker
from teams import TeamClassifier, TeamVoter, torso_descriptor, NEAR as TEAM_NEAR
from pipeline import (
    FEATURE_COLS,
    HAS_SUPERVISION,
    HAS_ULTRALYTICS,
    NET_Y_CM,
    SEQ_LEN,
    BallTracker,
    _compute_frame_features,
    _extract_positions,
    _filter_players_by_team_side,
    _team_labels_geometric,
)


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
    detector,
    team_side: str,
    net_margin_cm: float,
    manual_corners=None,
    tracker_type: str = "bytetrack",
    q_scale: float = 4.0,
    with_reid: bool = False,
    team_split: str = "colour",
    auto_court: bool = False,
    court_coords: str = "linear",
) -> np.ndarray | None:
    """Extract a fixed-length (SEQ_LEN, len(FEATURE_COLS)) sequence from one
    clip using the SAME per-frame steps as pipeline.run_pipeline: abstract
    detector -> court-mask filter -> tracker adapter -> team split ->
    slot-stable position extraction -> frame features. Keeping the two code
    paths identical prevents a train/serve feature mismatch.

    TWO-PHASE for the colour team split
    -----------------------------------
    The colour model needs court-side statistics over many detections before it
    can label a cluster, but a clip is only SEQ_LEN frames long - a streaming
    warm-up would never finish inside one clip. So phase A runs the expensive
    part once (detect -> mask -> track) and banks the per-frame boxes; the
    colour model is then fitted on the whole clip at once and phase B replays
    the banked frames to build features. Detection still runs exactly once per
    frame, and the resulting labels are strictly better than a streaming
    warm-up because they use the entire clip's evidence.
    """
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        return None

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    calibrator = CourtCalibrator(
        frame_w, frame_h,
        manual_corners=manual_corners,
        auto=auto_court,
        force_linear=(court_coords == "linear"),
    )
    tracker = create_tracker(tracker_type, fps=fps, q_scale=q_scale,
                             with_reid=with_reid)
    ball_tracker = BallTracker()
    slot_map = SlotManager()   # occlusion-bridged persistent slots

    use_colour = (team_split == "colour" and team_side != "all")

    # ── Phase A: detect -> court mask -> track, banking every frame ─────────
    banked: list[dict] = []
    desc_all: list[np.ndarray] = []
    ys_all: list[float] = []

    while len(banked) < SEQ_LEN:
        ret, frame = cap.read()
        if not ret:
            break

        calibrator.update(frame)
        det = detector.predict(frame)

        person_xyxy, person_conf = calibrator.filter_detections(
            det.person_xyxy, det.person_conf)

        if not use_colour:
            person_xyxy, person_conf = _filter_players_by_team_side(
                person_xyxy, person_conf, calibrator, team_side, net_margin_cm)

        det.person_xyxy, det.person_conf = person_xyxy, person_conf
        tracked_xyxy, track_ids = tracker.update(det, frame)
        tracked_ball, _ = ball_tracker.update(det.ball_center)

        court_ys = [
            calibrator.pixel_to_court(float((x1 + x2) / 2.0), float(y2))[1]
            for (x1, y1, x2, y2) in tracked_xyxy
        ]
        rec = {"xyxy": tracked_xyxy, "ids": track_ids,
               "ball": tracked_ball, "court_ys": court_ys}
        if use_colour and len(tracked_xyxy):
            d = torso_descriptor(frame, tracked_xyxy)
            rec["desc"] = d
            for row, y in zip(d, court_ys):
                if np.isfinite(y):
                    desc_all.append(row)
                    ys_all.append(float(y))
        banked.append(rec)

    cap.release()

    # ── Phase B: fit the colour model on the whole clip, then build features ─
    centres = cluster_team = None
    if use_colour and len(desc_all) >= 12:
        clf = TeamClassifier(warmup_frames=0, min_samples=0,
                             net_y_cm=NET_Y_CM)
        clf._desc, clf._ys = desc_all, ys_all      # whole-clip evidence
        clf._fit()
        if clf.degenerate:
            # Every cluster landed on one side: the population is not two
            # teams (usually the crowd, when no court mask is active). Trusting
            # it would extract a clip with zero players, silently.
            centres = cluster_team = None
        else:
            centres, cluster_team = clf.centres, clf.cluster_team

    voter = TeamVoter()
    want = TEAM_NEAR if team_side == "bottom" else TEAM_FAR
    rows: list[np.ndarray] = []

    for rec in banked:
        xyxy, ids = rec["xyxy"], rec["ids"]
        if use_colour and len(xyxy):
            if centres is not None and "desc" in rec:
                d2 = ((rec["desc"][:, None, :] - centres[None, :, :]) ** 2).sum(-1)
                labels = cluster_team[d2.argmin(1)]
            else:
                labels = _team_labels_geometric(rec["court_ys"], net_margin_cm)
            voted = voter.update(ids, labels)
            keep = voted == want
            xyxy, ids = xyxy[keep], ids[keep]

        player_positions, ball_pos = _extract_positions(
            xyxy, ids, rec["ball"], calibrator, slot_map)
        rows.append(_compute_frame_features(player_positions, ball_pos))

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

    manual_corners = args.court_corners
    if manual_corners is not None:
        print("Using supplied court homography for coordinate mapping.")
    else:
        print("No court corners supplied. Using linear pixel-to-court scaling.")

    clips = _iter_clip_files(dataset_dir)
    if not clips:
        print(f"[ERROR] No video clips found under: {dataset_dir}")
        sys.exit(1)

    if args.max_clips > 0:
        clips = clips[: args.max_clips]

    print(f"Loading detector model: {args.yolo_model}")
    if args.ball_model:
        print(f"Loading ball model    : {args.ball_model}")
    detector = create_detector(
        args.yolo_model,
        ball_weights=args.ball_model,
        conf_threshold=args.conf_threshold,
        person_class_id=args.person_class_id,
        ball_class_id=args.ball_class_id,
        imgsz=args.imgsz,
    )
    print(f"  Backend: {detector.name}")
    print(f"  Team split: {args.team_split}")
    print(f"  Detector classes: {detector.class_names} -> "
          f"person_id={detector.person_id} ball_id={detector.ball_id}")
    if detector.person_id is None:
        print("[ERROR] No person/player class found; pass --person-class-id.")
        return

    print(f"Preparing {len(clips)} clips -> {out_dir}")
    success = 0
    skipped = 0

    for i, (clip_path, folder_name, label_name) in enumerate(clips, start=1):
        seq = _extract_clip_sequence(
            clip_path=clip_path,
            detector=detector,
            team_side=args.team_side,
            net_margin_cm=args.net_margin_cm,
            manual_corners=manual_corners,
            tracker_type=args.tracker,
            q_scale=args.kalman_q_scale,
            with_reid=args.reid,
            team_split=args.team_split,
            auto_court=args.auto_court,
            court_coords=args.court_coords,
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
        default="yolo11n.pt",
        help="Detector weights path (default yolo11n.pt; pass your best.pt).",
    )
    parser.add_argument(
        "--ball-model", default=None,
        help="Separate weights used ONLY for the ball. MUST match what "
             "pipeline.py will use at inference time, or training and serving "
             "see different feature distributions.",
    )
    parser.add_argument(
        "--imgsz", type=int, default=DEFAULT_IMGSZ,
        help=f"Detector inference resolution (default {DEFAULT_IMGSZ}). MUST "
             "match pipeline.py.",
    )
    parser.add_argument(
        "--auto-court", action="store_true",
        help="Detect the court quad and use it as an in/out-of-court MASK "
             "(drops crowd, bench, coaches). MUST match pipeline.py: without "
             "it the crowd enters the population, and on clips where the crowd "
             "outnumbers the players it captures the colour clusters and the "
             "clip extracts with zero players.",
    )
    parser.add_argument(
        "--court-coords", choices=("linear", "homography"), default="linear",
        help="Coordinate mapping. 'linear' keeps the pixel->court scaling the "
             "existing CSVs were built with while still letting --auto-court "
             "clean the detection population. MUST match pipeline.py.",
    )
    parser.add_argument(
        "--team-split", choices=("colour", "geometric"), default="colour",
        help="Team separation strategy. MUST match pipeline.py: the two teams "
             "produce different formations, so a mismatch here is a silent "
             "train/serve feature mismatch.",
    )
    parser.add_argument(
        "--person-class-id", type=int, default=None,
        help="Override person/player class id (default: auto-detect by name).",
    )
    parser.add_argument(
        "--ball-class-id", type=int, default=None,
        help="Override ball class id (default: auto-detect by name).",
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
        help="Delete existing CSV files in the output directory before writing.",
    )
    parser.add_argument(
        "--tracker",
        choices=("bytetrack", "botsort"),
        default="bytetrack",
        help="Tracking backend for extraction (botsort adds CMC + optional Re-ID).",
    )
    parser.add_argument(
        "--kalman-q-scale",
        type=float,
        default=4.0,
        help="Kalman process-noise multiplier (explosive-kinetics tuning).",
    )
    parser.add_argument(
        "--reid",
        action="store_true",
        help="Enable appearance Re-ID in BoT-SORT.",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=0,
        help="Process at most N clips (0 = all). Useful for smoke tests.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(_parse_args())
