"""
diagnose_video.py - ONE run that prints the truth at EVERY pipeline stage.

Run on the laptop:
    .venv\\Scripts\\python fyp\\diagnose_video.py
It processes 3 frames from the demo segment and reports, per frame:
raw detections per class -> court-quad detection -> mask survivors ->
side-filter survivors -> tracker output. Saves diag_<frame>.jpg with every
stage drawn (RED raw persons, YELLOW post-mask, GREEN post-side-filter,
CYAN court quad, BLUE ball, WHITE assumed net line).
"""
from __future__ import annotations

import hashlib
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import cv2  # noqa: E402
import numpy as np  # noqa: E402


def main():
    video = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "videoplayback (4).mp4")
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 11290
    # Players from stock COCO weights, ball from the volleyball fine-tune
    # (technical review §13: the fine-tune scores 0.06 max on players in this
    # video, stock scores 0.91 - so the player source MUST be the stock model).
    weights = sys.argv[3] if len(sys.argv) > 3 else "yolo11n.pt"
    ball_weights = os.path.join(HERE, "volleyball_best.pt")

    print("=" * 70)
    print("ENVIRONMENT")
    print("  python     :", sys.version.split()[0])
    for mod in ("torch", "ultralytics", "supervision", "cv2"):
        try:
            m = __import__(mod)
            print(f"  {mod:<11}:", getattr(m, "__version__", "?"))
        except Exception as e:
            print(f"  {mod:<11}: IMPORT FAILED -> {e}")
    for tag, path in (("player wts", weights), ("ball wts  ", ball_weights)):
        if os.path.exists(path):
            with open(path, "rb") as fh:
                md5 = hashlib.md5(fh.read()).hexdigest()
            print(f"  {tag} :", path)
            print(f"  {'md5':<11}:", md5, "| size:", os.path.getsize(path))
        else:
            print(f"  {tag} :", path, "(auto-download)")

    from interfaces import create_detector
    from tracking import create_tracker
    from court import detect_court_quad, CourtCalibrator, NET_Y_CM

    det = create_detector(weights, ball_weights=ball_weights, conf_threshold=0.35)
    print("  backend    :", det.name)
    print("  class map  :", det.class_names)
    print("  person_id  :", det.person_id, "| ball_id:", det.ball_id)

    cap = cv2.VideoCapture(video)
    print("  video open :", cap.isOpened(), "| frames:",
          int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    h_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    tracker = create_tracker("bytetrack", fps=30.0)
    print("  tracker    :", tracker.name)

    for fidx in (start, start + 150, start + 300):
        print("-" * 70)
        print(f"FRAME {fidx}")
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ok, frame = cap.read()
            if not ok:
                print("  READ FAILED"); continue
            print(f"  decoded: shape={frame.shape} mean={frame.mean():.1f} std={frame.std():.1f}")

            # 1. RAW detector output
            d = det.predict(frame)
            print(f"  RAW persons: {len(d.person_xyxy)}  conf={np.round(d.person_conf, 2).tolist()[:10]}")
            print(f"  RAW ball   : {'YES ' + str(np.round(d.ball_center,1).tolist()) if d.ball_center is not None else 'no'}")

            viz = frame.copy()
            for (x1, y1, x2, y2) in d.person_xyxy.astype(int):
                cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 0, 255), 2)      # RED raw
            if d.ball_center is not None:
                cv2.circle(viz, tuple(d.ball_center.astype(int)), 8, (255, 0, 0), 2)

            # 2. court quad + mask
            quad = detect_court_quad(frame)
            cal = CourtCalibrator(w_frame, h_frame, auto=True, force_linear=True)
            if quad is not None:
                cal.set_auto_quad(quad)
                print(f"  quad: {quad.round(0).tolist()}")
                cv2.polylines(viz, [quad.astype(np.int32)], True, (255, 255, 0), 2)  # CYAN
            else:
                print("  quad: DETECTION FAILED (mask = pass-through)")
            mx, mc = cal.filter_detections(d.person_xyxy, d.person_conf)
            print(f"  after MASK : {len(mx)} persons")
            for (x1, y1, x2, y2) in mx.astype(int):
                cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 255, 255), 2)    # YELLOW

            # 3. side filter (linear coords, bottom, margin 30)
            kept, feet = [], []
            for i, (x1, y1, x2, y2) in enumerate(mx):
                px, py = float((x1 + x2) / 2), float(y2)
                _, cy = cal.pixel_to_court(px, py)
                feet.append((round(px), round(py), round(cy)))
                if cy >= NET_Y_CM + 30.0:
                    kept.append(i)
            print(f"  feet(px,py)->court_y: {feet}")
            print(f"  after SIDE FILTER (need y>=480): {len(kept)} persons")
            for i in kept:
                x1, y1, x2, y2 = mx[i].astype(int)
                cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 255, 0), 3)      # GREEN

            net_py = int(NET_Y_CM / 900.0 * h_frame)
            cv2.line(viz, (0, net_py), (w_frame, net_py), (255, 255, 255), 1)

            # 4. tracker on the survivors
            d.person_xyxy = mx[kept] if kept else np.empty((0, 4))
            d.person_conf = mc[kept] if kept else np.empty((0,))
            txyxy, tids = tracker.update(d, frame)
            print(f"  TRACKER out: {len(txyxy)} tracks ids={tids.tolist() if len(tids) else []}")

            out = os.path.join(ROOT, f"diag_{fidx}.jpg")
            cv2.imwrite(out, viz)
            print("  saved:", out)
        except Exception:
            traceback.print_exc()
    cap.release()
    print("=" * 70)
    print("Send diagnose_output.txt and the diag_*.jpg files back for analysis.")


if __name__ == "__main__":
    main()
