import os
import subprocess
import shutil
from datetime import timedelta

import pandas as pd


CANONICAL_CLASSES = {
    "coordinated_attack": "Coordinated_Attack",
    "coordinated_defense": "Coordinated_Defense",
    "coordinated_defence": "Coordinated_Defense",
    "delayed_support": "Delayed_Support",
    "spacing_breakdown": "Spacing_Breakdown",
}


def _parse_time_to_seconds(anchor_time_str: str) -> float:
    parts = anchor_time_str.strip().split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid time format: {anchor_time_str}")

    hh = int(parts[0])
    mm = int(parts[1])
    sec = float(parts[2])
    return hh * 3600 + mm * 60 + sec


def _format_seconds_for_ffmpeg(total_seconds: float) -> str:
    total_seconds = max(0.0, total_seconds)
    td = timedelta(seconds=total_seconds)
    total = td.total_seconds()
    hh = int(total // 3600)
    mm = int((total % 3600) // 60)
    ss = total % 60
    return f"{hh:02d}:{mm:02d}:{ss:06.3f}"


def _normalize_tactic(raw_tactic: str) -> str:
    key = raw_tactic.strip().lower().replace(" ", "_")
    return CANONICAL_CLASSES.get(key, raw_tactic.strip())


def _resolve_ffmpeg_executable() -> str | None:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path

    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def build_dataset_pipeline(labels_path: str = "labels.csv") -> None:
    print("Initializing Dataset Extraction Pipeline...\n")

    ffmpeg_exe = _resolve_ffmpeg_executable()
    if not ffmpeg_exe:
        print(
            "Error: ffmpeg is not available. Install ffmpeg or run `pip install imageio-ffmpeg`."
        )
        return

    labels_path = os.path.abspath(labels_path)
    if not os.path.exists(labels_path):
        print(f"Error: {labels_path} not found.")
        return

    root_dir = os.path.dirname(labels_path)
    dataset_dir = os.path.join(root_dir, "dataset")

    df = pd.read_csv(labels_path)
    required_cols = {"video_file", "anchor_time", "tactic_class"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"Error: labels.csv is missing columns: {sorted(missing)}")
        return

    for class_name in sorted(set(CANONICAL_CLASSES.values())):
        os.makedirs(os.path.join(dataset_dir, class_name), exist_ok=True)

    print(f"Verified folder structure in {dataset_dir}")

    success_count = 0
    skip_count = 0

    for index, row in df.iterrows():
        raw_video_file = str(row["video_file"]).strip()
        anchor_time = str(row["anchor_time"]).strip()
        raw_tactic = str(row["tactic_class"]).strip()
        tactic = _normalize_tactic(raw_tactic)

        if tactic not in CANONICAL_CLASSES.values():
            print(f"Warning: Unknown tactic '{raw_tactic}' on row {index + 1}. Skipping.")
            skip_count += 1
            continue

        if os.path.isabs(raw_video_file):
            video_path = raw_video_file
        else:
            video_path = os.path.join(root_dir, raw_video_file)

        if not os.path.exists(video_path):
            print(f"Warning: video not found '{video_path}' for row {index + 1}. Skipping.")
            skip_count += 1
            continue

        try:
            start_seconds = _parse_time_to_seconds(anchor_time)
        except ValueError as exc:
            print(f"Warning: {exc} on row {index + 1}. Skipping.")
            skip_count += 1
            continue

        if tactic == "Coordinated_Defense":
            start_seconds -= 0.5

        start_str = _format_seconds_for_ffmpeg(start_seconds)
        clip_name = f"{os.path.splitext(os.path.basename(raw_video_file))[0]}_clip_{index + 1:03d}.mp4"
        clip_path = os.path.join(dataset_dir, tactic, clip_name)

        cmd = [
            ffmpeg_exe,
            "-y",
            "-ss",
            start_str,
            "-i",
            video_path,
            "-t",
            "00:00:01.000",
            "-c",
            "copy",
            clip_path,
        ]

        print(f"[{index + 1}/{len(df)}] {tactic} at {start_str} -> {clip_path}")
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if result.returncode == 0:
            success_count += 1
        else:
            print(f"Warning: ffmpeg failed on row {index + 1}.")
            skip_count += 1

    print(f"\nPipeline complete. Extracted: {success_count}, Skipped: {skip_count}")


if __name__ == "__main__":
    build_dataset_pipeline()