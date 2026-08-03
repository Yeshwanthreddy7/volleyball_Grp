"""
verify_detector_dataset.py - strict pre-training gate for the YOLO dataset.

Replicates the EXACT acceptance rules Ultralytics applies in
`utils/dataset.py::verify_image_label` for the `detect` task, without needing
torch, so a broken dataset is caught BEFORE a (paid/slow) training run:

  R1  every image decodes (cv2) and is >= 10x10 px
  R2  every image has a label file and vice versa (stem pairing)
  R3  every label line: first token is an int class id, 0 <= id < nc
  R4  box lines have exactly 4 coords; polygon lines >= 6 coords, even count
      (Ultralytics converts polygons to boxes via min/max for `detect`)
  R5  all coordinates within [0, 1]
  R6  the box derived from each line has width > 0 and height > 0
  R7  data.yaml: `path` + train/val dirs exist, `nc` matches `names`

Exit code 0 = dataset will be fully accepted by the trainer, no silently
dropped labels. Usage:
  python fyp/verify_detector_dataset.py "volleyball-detection.yolov11"
"""
from __future__ import annotations

import os
import re
import sys

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def check_label_file(path, nc):
    """Return (instances_per_class, errors:list[str]) for one label file."""
    counts, errors = {}, []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for ln, raw in enumerate(fh, 1):
            tok = raw.split()
            if not tok:
                continue
            try:
                cls = int(float(tok[0]))
                vals = [float(x) for x in tok[1:]]
            except ValueError:
                errors.append("line {}: non-numeric token".format(ln))
                continue
            if not 0 <= cls < nc:                                   # R3
                errors.append("line {}: class {} outside 0..{}".format(ln, cls, nc - 1))
                continue
            if len(vals) == 4:
                xc, yc, w, h = vals
                xs, ys = [xc - w / 2, xc + w / 2], [yc - h / 2, yc + h / 2]
            elif len(vals) >= 6 and len(vals) % 2 == 0:             # R4
                xs, ys = vals[0::2], vals[1::2]
            else:
                errors.append("line {}: {} coords (need 4 or even >=6)".format(ln, len(vals)))
                continue
            if any(v < 0.0 or v > 1.0 for v in vals):               # R5
                errors.append("line {}: coordinate outside [0,1]".format(ln))
                continue
            if max(xs) - min(xs) <= 0 or max(ys) - min(ys) <= 0:    # R6
                errors.append("line {}: degenerate box (w or h == 0)".format(ln))
                continue
            counts[cls] = counts.get(cls, 0) + 1
    return counts, errors


def verify_split(root, split, nc, decode_images=True):
    img_d = os.path.join(root, split, "images")
    lbl_d = os.path.join(root, split, "labels")
    report = {"images": 0, "instances": {}, "errors": []}
    if not os.path.isdir(img_d):
        report["errors"].append("missing dir: " + img_d)
        return report
    cv2 = None
    if decode_images:
        try:
            import cv2 as _cv2
            cv2 = _cv2
        except ImportError:
            report["errors"].append("cv2 unavailable - skipped image decode check")
    stems = set()
    for f in sorted(os.listdir(img_d)):
        stem, ext = os.path.splitext(f)
        if ext.lower() not in IMG_EXTS:
            continue
        report["images"] += 1
        stems.add(stem)
        if cv2 is not None:
            im = cv2.imread(os.path.join(img_d, f))
            if im is None or im.shape[0] < 10 or im.shape[1] < 10:  # R1
                report["errors"].append("undecodable/tiny image: " + f)
        lp = os.path.join(lbl_d, stem + ".txt")
        if not os.path.exists(lp):                                  # R2
            report["errors"].append("no label file for image: " + f)
            continue
        counts, errs = check_label_file(lp, nc)
        for c, k in counts.items():
            report["instances"][c] = report["instances"].get(c, 0) + k
        report["errors"] += ["{}: {}".format(stem, e) for e in errs]
    if os.path.isdir(lbl_d):                                        # R2 reverse
        for f in os.listdir(lbl_d):
            if f.endswith(".txt") and os.path.splitext(f)[0] not in stems:
                report["errors"].append("orphan label (no image): " + f)
    return report


def verify_yaml(root):
    """R7 - returns (nc, names, errors)."""
    errors = []
    yp = os.path.join(root, "data.yaml")
    if not os.path.exists(yp):
        return 2, ["ball", "player"], ["data.yaml missing"]
    text = open(yp, encoding="utf-8").read()
    m = re.search(r"(?m)^nc\s*:\s*(\d+)", text)
    nc = int(m.group(1)) if m else 2
    names = re.findall(r"(?m)^\s+\d+\s*:\s*(\S+)", text)
    if not names:
        m2 = re.search(r"names:\s*(\[[^\]]*\])", text)
        if m2:
            import ast
            names = [str(x) for x in ast.literal_eval(m2.group(1))]
    if names and len(names) != nc:
        errors.append("nc={} but {} names".format(nc, len(names)))
    for key in ("train", "val"):
        m3 = re.search(r"(?m)^{}\s*:\s*(.+)$".format(key), text)
        if not m3:
            errors.append("data.yaml missing key: " + key)
            continue
        rel = m3.group(1).strip()
        if not os.path.isdir(os.path.join(root, rel)):
            errors.append("{}: {} does not exist under {}".format(key, rel, root))
    return nc, names or ["ball", "player"], errors


def main(argv=None):
    root = os.path.abspath(argv[0] if argv else
                           (sys.argv[1] if len(sys.argv) > 1
                            else "volleyball-detection.yolov11"))
    decode = "--no-decode" not in (argv or sys.argv)
    nc, names, yerr = verify_yaml(root)
    print("dataset : {}".format(root))
    print("classes : nc={} names={}".format(nc, names))
    all_errors = list(yerr)
    for split in ("train", "valid"):
        r = verify_split(root, split, nc, decode_images=decode)
        inst = ", ".join("{}={}".format(names[c] if c < len(names) else c, k)
                         for c, k in sorted(r["instances"].items()))
        print("{:<6}: {} images | {} | {} error(s)".format(
            split, r["images"], inst or "-", len(r["errors"])))
        all_errors += r["errors"]
    if all_errors:
        print("\nERRORS ({}):".format(len(all_errors)))
        for e in all_errors[:40]:
            print("  - " + e)
        if len(all_errors) > 40:
            print("  ... and {} more".format(len(all_errors) - 40))
        print("\nVERDICT: FAIL - fix before training.")
        return 1
    print("\nVERDICT: PASS - every image and label satisfies the trainer's rules;")
    print("nothing will be silently dropped at train time.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
