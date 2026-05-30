"""
process_video.py – Input-to-output video converter for volleyball analytics.

Reads an input MP4, annotates each frame with:
  • Moving-object detection (players / ball) via MOG2 background subtraction
  • Bounding boxes drawn around detected moving regions
  • A court-net reference line at the vertical midpoint of the frame
  • Frame number and timestamp overlay

Usage:
    python process_video.py <input_video> <output_video>

Example:
    python process_video.py test1.mp4 output.mp4
"""

import argparse
import sys

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Constants / tuneable parameters
# ---------------------------------------------------------------------------
# Background subtractor – history frames and sensitivity
BG_HISTORY = 120          # frames used to build background model
BG_VAR_THRESHOLD = 50     # lower → more sensitive to motion
BG_DETECT_SHADOWS = False  # skip shadow detection for speed

# Morphological cleanup of the foreground mask
MORPH_KERNEL_SIZE = (5, 5)
DILATE_ITERATIONS = 3

# Minimum contour area (px²) to be considered a real object (not noise)
MIN_CONTOUR_AREA = 400

# Maximum contour area (px²) – ignore very large blobs (e.g. full-court noise)
MAX_CONTOUR_AREA = 80_000

# Visual style
BOX_COLOR = (0, 255, 0)       # bright green bounding boxes
BOX_THICKNESS = 2
NET_COLOR = (0, 165, 255)     # orange net line
NET_THICKNESS = 2
TEXT_COLOR = (255, 255, 255)  # white text
TEXT_SCALE = 0.7
TEXT_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Output codec
FOURCC = cv2.VideoWriter_fourcc(*"mp4v")


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_video(input_path: str, output_path: str) -> None:
    """Read *input_path*, annotate each frame, write annotated video to *output_path*."""

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {input_path}", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Input : {input_path}")
    print(f"Size  : {width}x{height}  FPS: {fps:.2f}  Frames: {total_frames}")
    print(f"Output: {output_path}")

    writer = cv2.VideoWriter(output_path, FOURCC, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        print(f"[ERROR] Cannot create output video: {output_path}", file=sys.stderr)
        sys.exit(1)

    # Background subtractor – learns the court surface, highlights players/ball
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=BG_HISTORY,
        varThreshold=BG_VAR_THRESHOLD,
        detectShadows=BG_DETECT_SHADOWS,
    )

    # Morphological kernel for mask cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL_SIZE)

    # Court net reference line at the vertical midpoint of the frame
    net_y = height // 2

    frame_idx = 0
    print("Processing", end="", flush=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % 300 == 0:
            print(f" {frame_idx}/{total_frames}", end="", flush=True)

        # ── Background subtraction ──────────────────────────────────────────
        fg_mask = bg_subtractor.apply(frame)

        # Clean up noise with morphological operations
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=DILATE_ITERATIONS)

        # ── Find contours of moving objects ─────────────────────────────────
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        annotated = frame.copy()

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (MIN_CONTOUR_AREA <= area <= MAX_CONTOUR_AREA):
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), BOX_COLOR, BOX_THICKNESS)

        # ── Court net reference line ─────────────────────────────────────────
        cv2.line(annotated, (0, net_y), (width, net_y), NET_COLOR, NET_THICKNESS)
        cv2.putText(
            annotated, "NET", (10, net_y - 8),
            FONT, TEXT_SCALE * 0.8, NET_COLOR, TEXT_THICKNESS,
        )

        # ── Frame counter / timestamp overlay ───────────────────────────────
        timestamp = frame_idx / fps
        info_text = f"Frame {frame_idx:05d}  |  {timestamp:.2f}s"
        cv2.putText(
            annotated, info_text, (10, 30),
            FONT, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS,
        )

        writer.write(annotated)

    cap.release()
    writer.release()
    print(f"\nDone. {frame_idx} frames written to '{output_path}'.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an input volleyball video into an annotated output video.",
    )
    parser.add_argument("input", help="Path to the input MP4 video file.")
    parser.add_argument("output", help="Path for the output annotated MP4 video.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    process_video(args.input, args.output)
