"""
train_detector.py - fine-tune YOLOv11 on the volleyball Roboflow export and
verify the result end-to-end.

Run fyp/prepare_detector_dataset.py FIRST (creates the leak-free valid/ split
and a correct data.yaml). Then:

  Full training (Colab / GPU):
      python fyp/train_detector.py --epochs 100 --imgsz 640
  CPU laptop (overnight):
      python fyp/train_detector.py --epochs 100 --imgsz 640 --batch 8
  Smoke test (proves the whole chain, ~minutes on CPU):
      python fyp/train_detector.py --smoke

What it guarantees after training:
  1. best.pt loads and its class map is printed.
  2. detect_utils.resolve_class_ids() finds person+ball ids BY NAME
     (guards the Roboflow {0:'ball',1:'player'} vs COCO {0:'person'} loophole).
  3. Validation metrics are computed on the held-out temporal split.
  4. A real validation image is run through predict() without exceptions.

It also re-patches the data.yaml `path:` to THIS machine's absolute dataset
location, so the dataset folder can be moved/zipped/unzipped freely.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from detect_utils import resolve_class_ids  # noqa: E402  (torch-free)


def patch_yaml_path(yaml_path):
    """Rewrite/insert the `path:` key with the yaml's own absolute directory."""
    root = os.path.dirname(os.path.abspath(yaml_path)).replace("\\", "/")
    with open(yaml_path, "r", encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    out, done = [], False
    for ln in lines:
        if re.match(r"^path\s*:", ln):
            out.append("path: " + root)
            done = True
        else:
            out.append(ln)
    if not done:
        out.insert(0, "path: " + root)
    with open(yaml_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    return root


def first_val_image(dataset_root):
    d = os.path.join(dataset_root, "valid", "images")
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            if os.path.splitext(f)[1].lower() in (".jpg", ".jpeg", ".png"):
                return os.path.join(d, f)
    return None


def verify_weights(weights, dataset_root, imgsz):
    """Load best.pt, check class-id resolution, run one real prediction."""
    from ultralytics import YOLO
    model = YOLO(weights)
    names = model.names
    print("\n[verify] class map of {}: {}".format(os.path.basename(weights), names))
    person_id, ball_id = resolve_class_ids(names)
    print("[verify] resolve_class_ids -> person={}, ball={}".format(person_id, ball_id))
    if person_id is None:
        print("[verify] FAIL: no person/player class resolvable by name. "
              "Use --person-class-id downstream, or fix dataset names.")
        return False
    img = first_val_image(dataset_root)
    if img is None:
        print("[verify] WARN: no validation image found for the predict check.")
        return True
    res = model.predict(img, imgsz=imgsz, conf=0.25, verbose=False)[0]
    cls = res.boxes.cls.tolist() if res.boxes is not None else []
    print("[verify] predict on {}: {} detections "
          "({} player, {} ball) - no exceptions.".format(
              os.path.basename(img), len(cls),
              sum(1 for c in cls if int(c) == person_id),
              sum(1 for c in cls if ball_id is not None and int(c) == ball_id)))
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Fine-tune YOLOv11 on the volleyball dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data",
                    default=os.path.join(PROJECT_ROOT,
                                         "volleyball-detection.yolov11",
                                         "data.yaml"),
                    help="Path to data.yaml (after prepare_detector_dataset.py).")
    ap.add_argument("--model", default="yolo11n.pt",
                    help="Base weights to fine-tune (auto-downloads).")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640,
                    help="640 minimum - the ball is a tiny object.")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=25,
                    help="Early-stopping patience (epochs without val improvement).")
    ap.add_argument("--device", default=None,
                    help="'0' GPU, 'cpu', or leave unset for auto.")
    ap.add_argument("--workers", type=int, default=2,
                    help="Dataloader workers (keep small on Windows/CPU).")
    ap.add_argument("--project", default=os.path.join(PROJECT_ROOT, "runs"))
    ap.add_argument("--name", default="volleyball_yolo11")
    ap.add_argument("--smoke", action="store_true",
                    help="2 epochs @ 320 px, batch 8 - chain validation only.")
    ap.add_argument("--verify-only", default=None, metavar="BEST_PT",
                    help="Skip training; just verify an existing weights file.")
    args = ap.parse_args(argv)

    if not os.path.exists(args.data):
        print("ERROR: {} not found. Run fyp/prepare_detector_dataset.py first."
              .format(args.data))
        return 1
    dataset_root = patch_yaml_path(args.data)
    print("[setup] data.yaml path patched -> {}".format(dataset_root))
    if not os.path.isdir(os.path.join(dataset_root, "valid", "images")):
        print("ERROR: no valid/ split. Run fyp/prepare_detector_dataset.py first.")
        return 1

    if args.verify_only:
        return 0 if verify_weights(args.verify_only, dataset_root, args.imgsz) else 1

    if args.smoke:
        args.epochs, args.imgsz, args.batch = 2, 320, 8
        print("[setup] SMOKE MODE: epochs=2 imgsz=320 batch=8")

    from ultralytics import YOLO
    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        seed=0,             # reproducible for the report
        cos_lr=True,
        plots=True,
        exist_ok=True,
    )
    save_dir = str(getattr(results, "save_dir", os.path.join(args.project, args.name)))
    best = os.path.join(save_dir, "weights", "best.pt")
    print("\n[done] best weights: {}".format(best))

    ok = verify_weights(best, dataset_root, max(args.imgsz, 640))
    print("\n" + "=" * 68)
    print("NEXT STEPS (train/serve consistency - do not skip):")
    print("  1. copy {} -> fyp/volleyball_best.pt".format(best))
    print("  2. re-extract training CSVs with the SAME detector:")
    print('     python fyp/prepare_training_data.py dataset --output-dir '
          'training_csv --yolo-model fyp/volleyball_best.pt --clean-output')
    print("  3. retrain the Mamba on the re-extracted CSVs:")
    print("     python fyp/train_mamba.py training_csv --augment --epochs 80 "
          "--checkpoint mamba_checkpoint_v2.pt")
    print("  4. run the full pipeline with the custom detector:")
    print('     python fyp/pipeline.py <match.mp4> --yolo-model '
          'fyp/volleyball_best.pt --tracker botsort --auto-court')
    print("=" * 68)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
