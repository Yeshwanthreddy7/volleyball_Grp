"""
prepare_detector_dataset.py - make a Roboflow YOLO export trainable, leak-free.

Why this exists
---------------
A Roboflow "download dataset" export often ships with ONLY a train/ split and a
data.yaml whose relative paths ('../train/images') break the moment the folder
moves. Worse, the images are consecutive frames sampled from the SAME video, so
a naive random split leaks near-duplicate frames into validation and inflates
mAP - the first thing a CV examiner checks.

What it does (stdlib only - no torch/cv2 - unit-testable anywhere)
  1. Audits every label file: per-class instance counts, polygon vs box lines,
     malformed lines, out-of-range coordinates, image/label pairing.
     (Roboflow polygon labels are fine: Ultralytics converts polygons to boxes
     automatically for the `detect` task.)
  2. Splits train/ into train/ + valid/ using CONTIGUOUS FRAME BLOCKS spread
     across the timeline, so validation frames come from different moments of
     play than training frames (temporal split, not random).
  3. Excludes a small boundary "gap" around each validation block so no
     adjacent (near-identical) frame straddles the split.
  4. Guarantees the minority class (ball) is present in validation.
  5. Rewrites data.yaml with an absolute `path:` plus train/valid keys.

Usage
  python fyp/prepare_detector_dataset.py "volleyball-detection.yolov11"
  python fyp/prepare_detector_dataset.py "volleyball-detection.yolov11" --dry-run
  python fyp/prepare_detector_dataset.py "volleyball-detection.yolov11" --force
"""
from __future__ import annotations

import argparse
import ast
import os
import re
import shutil
import sys

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested)                                                  #
# --------------------------------------------------------------------------- #
def parse_frame_number(stem):
    """'frame_0123_jpg.rf.HASH' -> 123. Falls back to first integer, else None."""
    m = re.search(r"frame[_-]?(\d+)", stem, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)", stem)
    return int(m.group(1)) if m else None


def audit_label_file(path):
    """Return {'counts': {cls: n}, 'poly': n, 'box': n, 'bad': n} for one file."""
    counts, poly, box, bad = {}, 0, 0, 0
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            tok = raw.split()
            if not tok:
                continue
            try:
                cls = int(float(tok[0]))
                vals = [float(x) for x in tok[1:]]
            except ValueError:
                bad += 1
                continue
            if any(v < -0.001 or v > 1.001 for v in vals):
                bad += 1
                continue
            if len(vals) == 4:
                box += 1
            elif len(vals) >= 6 and len(vals) % 2 == 0:
                poly += 1                    # polygon (segmentation) line
            else:
                bad += 1
                continue
            counts[cls] = counts.get(cls, 0) + 1
    return {"counts": counts, "poly": poly, "box": box, "bad": bad}


def plan_split(stems, val_fraction=0.15, n_blocks=12):
    """Partition stems (sorted by frame number) into contiguous blocks and pick
    evenly spaced blocks for validation.

    Returns (chunks, val_idx): chunks is a list of lists of stems in temporal
    order; val_idx the sorted indices of validation blocks. Deterministic.
    """
    ordered = sorted(
        stems,
        key=lambda s: (parse_frame_number(s) if parse_frame_number(s) is not None
                       else float("inf"), s),
    )
    n = len(ordered)
    if n == 0:
        return [], []
    n_blocks = max(2, min(n_blocks, n))
    bounds = [round(i * n / n_blocks) for i in range(n_blocks + 1)]
    chunks = [ordered[bounds[i]:bounds[i + 1]] for i in range(n_blocks)]
    n_val = max(1, int(round(val_fraction * n_blocks)))
    n_val = min(n_val, n_blocks - 1)
    step = n_blocks / n_val
    val_idx = sorted({min(n_blocks - 1, int((j + 0.5) * step)) for j in range(n_val)})
    return chunks, val_idx


def ensure_class_presence(chunks, val_idx, per_stem_counts, cls):
    """If class `cls` is absent from every validation block, swap the emptiest
    validation block for the training block richest in `cls`. Deterministic."""
    def blk(i):
        return sum(per_stem_counts.get(s, {}).get(cls, 0) for s in chunks[i])

    if any(blk(i) > 0 for i in val_idx):
        return list(val_idx)
    train_idx = [i for i in range(len(chunks)) if i not in set(val_idx)]
    if not train_idx:
        return list(val_idx)
    best_train = max(train_idx, key=lambda i: (blk(i), -i))
    if blk(best_train) == 0:
        return list(val_idx)                 # class absent everywhere
    worst_val = min(val_idx, key=lambda i: (blk(i), i))
    return sorted((set(val_idx) - {worst_val}) | {best_train})


def plan_gap_exclusions(ordered, val_set, gap=1):
    """Training stems within `gap` positions of any validation stem.

    These near-duplicate boundary frames are quarantined (used by NEITHER
    split) so no ~0.1 s-apart twin of a validation frame sits in training.
    """
    excluded = set()
    n = len(ordered)
    for k, s in enumerate(ordered):
        if s in val_set:
            continue
        lo, hi = max(0, k - gap), min(n - 1, k + gap)
        if any(ordered[j] in val_set for j in range(lo, hi + 1)):
            excluded.add(s)
    return excluded


def read_names_from_yaml(yaml_path):
    """Minimal parse of `nc:` and `names:` from a Roboflow data.yaml."""
    names = None
    try:
        with open(yaml_path, "r", encoding="utf-8") as fh:
            text = fh.read()
        m = re.search(r"names:\s*(\[[^\]]*\])", text)
        if m:
            names = [str(x) for x in ast.literal_eval(m.group(1))]
    except (OSError, ValueError, SyntaxError):
        pass
    return names or ["ball", "player"]


def render_data_yaml(dataset_root_abs, names):
    lines = [
        "# Rewritten by prepare_detector_dataset.py (leak-free temporal split).",
        "# `path:` is absolute for THIS machine; fyp/train_detector.py and the",
        "# Colab notebook re-patch it automatically after the folder moves.",
        "path: " + dataset_root_abs.replace("\\", "/"),
        "train: train/images",
        "val: valid/images",
        "",
        "nc: " + str(len(names)),
        "names:",
    ]
    lines += ["  {}: {}".format(i, n) for i, n in enumerate(names)]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Filesystem operations                                                       #
# --------------------------------------------------------------------------- #
def _pairs(images_dir, labels_dir):
    """Yield (stem, image_filename) for every image; label may be absent."""
    for f in sorted(os.listdir(images_dir)):
        stem, ext = os.path.splitext(f)
        if ext.lower() in IMG_EXTS:
            yield stem, f


def _move_pair(src_root, dst_root, split_from, split_to, stem, img_name):
    for kind, name in (("images", img_name), ("labels", stem + ".txt")):
        src = os.path.join(src_root, split_from, kind, name)
        dst_dir = os.path.join(dst_root, split_to, kind)
        if os.path.exists(src):
            os.makedirs(dst_dir, exist_ok=True)
            shutil.move(src, os.path.join(dst_dir, name))


def revert_split(root):
    """Move everything in valid/ and excluded_gap/ back into train/."""
    moved = 0
    for split in ("valid", "excluded_gap"):
        for kind in ("images", "labels"):
            d = os.path.join(root, split, kind)
            if not os.path.isdir(d):
                continue
            os.makedirs(os.path.join(root, "train", kind), exist_ok=True)
            for f in os.listdir(d):
                shutil.move(os.path.join(d, f),
                            os.path.join(root, "train", kind, f))
                moved += 1
        shutil.rmtree(os.path.join(root, split), ignore_errors=True)
    return moved


def summarize(root, split, names):
    img_d = os.path.join(root, split, "images")
    lbl_d = os.path.join(root, split, "labels")
    if not os.path.isdir(img_d):
        return None
    n_img, counts, poly, box, bad, unpaired = 0, {}, 0, 0, 0, 0
    for stem, _img in _pairs(img_d, lbl_d):
        n_img += 1
        lp = os.path.join(lbl_d, stem + ".txt")
        if not os.path.exists(lp):
            unpaired += 1
            continue
        a = audit_label_file(lp)
        poly, box, bad = poly + a["poly"], box + a["box"], bad + a["bad"]
        for c, k in a["counts"].items():
            counts[c] = counts.get(c, 0) + k
    per_cls = ", ".join("{}={}".format(names[c] if c < len(names) else c, k)
                        for c, k in sorted(counts.items()))
    return {"images": n_img, "per_class": per_cls, "poly": poly, "box": box,
            "bad": bad, "unpaired": unpaired}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("dataset", nargs="?", default="volleyball-detection.yolov11",
                    help="Dataset root (contains train/ and data.yaml).")
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--blocks", type=int, default=12,
                    help="Number of contiguous temporal blocks.")
    ap.add_argument("--gap", type=int, default=1,
                    help="Boundary frames quarantined around each val block.")
    ap.add_argument("--dry-run", action="store_true", help="Plan only; move nothing.")
    ap.add_argument("--force", action="store_true",
                    help="Revert any existing split first, then re-split.")
    args = ap.parse_args(argv)

    root = os.path.abspath(args.dataset)
    train_img = os.path.join(root, "train", "images")
    train_lbl = os.path.join(root, "train", "labels")
    if not os.path.isdir(train_img):
        print("ERROR: {} not found - point me at the Roboflow export root."
              .format(train_img))
        return 1

    names = read_names_from_yaml(os.path.join(root, "data.yaml"))

    valid_img = os.path.join(root, "valid", "images")
    if os.path.isdir(valid_img) and os.listdir(valid_img):
        if args.force:
            print("Existing split found - reverting {} files first."
                  .format(revert_split(root)))
        else:
            print("valid/ already exists and is non-empty. Use --force to re-split.")
            print(_report(root, names))
            return 0

    # ---- audit + plan -----------------------------------------------------
    stems, img_of, per_stem = [], {}, {}
    tot = {"poly": 0, "box": 0, "bad": 0, "unpaired": 0}
    for stem, img in _pairs(train_img, train_lbl):
        stems.append(stem)
        img_of[stem] = img
        lp = os.path.join(train_lbl, stem + ".txt")
        if os.path.exists(lp):
            a = audit_label_file(lp)
            per_stem[stem] = a["counts"]
            for k in ("poly", "box", "bad"):
                tot[k] += a[k]
        else:
            tot["unpaired"] += 1

    chunks, val_idx = plan_split(stems, args.val_fraction, args.blocks)
    # ball is the minority class - find its id by name
    ball_id = next((i for i, n in enumerate(names) if "ball" in n.lower()), 0)
    val_idx = ensure_class_presence(chunks, val_idx, per_stem, ball_id)
    val_set = {s for i in val_idx for s in chunks[i]}
    ordered = [s for c in chunks for s in c]
    gap_set = plan_gap_exclusions(ordered, val_set, args.gap)

    print("Audit: {} images | label lines: {} polygon, {} box, {} bad, "
          "{} image(s) without a label file".format(
              len(stems), tot["poly"], tot["box"], tot["bad"], tot["unpaired"]))
    print("Plan : {} blocks, val blocks {} -> {} val images, {} gap-quarantined, "
          "{} train".format(len(chunks), val_idx, len(val_set), len(gap_set),
                            len(stems) - len(val_set) - len(gap_set)))
    if args.dry_run:
        print("--dry-run: nothing moved.")
        return 0

    # ---- apply ------------------------------------------------------------
    for s in val_set:
        _move_pair(root, root, "train", "valid", s, img_of[s])
    for s in gap_set:
        _move_pair(root, root, "train", "excluded_gap", s, img_of[s])
    with open(os.path.join(root, "data.yaml"), "w", encoding="utf-8") as fh:
        fh.write(render_data_yaml(root, names))
    print("data.yaml rewritten (absolute path + train/valid keys).")
    print(_report(root, names))

    # ---- hard checks ------------------------------------------------------
    ok = True
    for split in ("train", "valid"):
        s = summarize(root, split, names)
        if not s or s["images"] == 0:
            print("FAIL: split '{}' is empty.".format(split))
            ok = False
        elif "ball" not in s["per_class"]:
            print("WARN: no ball instances in '{}'.".format(split))
    print("SPLIT OK - ready for training." if ok else "SPLIT FAILED.")
    return 0 if ok else 1


def _report(root, names):
    out = []
    for split in ("train", "valid", "excluded_gap"):
        s = summarize(root, split, names)
        if s:
            out.append("  {:<13} {:>4} images | {} | {} poly / {} box lines, "
                       "{} bad, {} unpaired".format(
                           split, s["images"], s["per_class"] or "-",
                           s["poly"], s["box"], s["bad"], s["unpaired"]))
    return "\n".join(out)


if __name__ == "__main__":
    sys.exit(main())
