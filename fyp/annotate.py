"""
annotate.py - blind human labelling of tactical clips, to break the circularity.

THE PROBLEM THIS SOLVES
-----------------------
Every label in `training_csv/` was written by `label_clips.py`, a threshold rule
engine. A model trained on those labels can only ever be measured against the
rules that produced them, so "94% accuracy" would mean "94% agreement with my
own heuristic" - a number that answers no question a examiner or a coach cares
about. Until human labels exist there is no ground truth in this project at all.

TWO DESIGN DECISIONS THAT MAKE THE RESULTING LABELS ADMISSIBLE
--------------------------------------------------------------
1. BLIND. The clips live in folders named after their rule label
   (`dataset/dataset/Coordinated_Attack/...`). Showing that name - or even
   presenting clips grouped by it - anchors the annotator to the rule engine's
   answer, and the measured agreement becomes self-fulfilling. This tool
   therefore shuffles clips across folders and never displays the folder, the
   filename, or the rule label. What it writes is an independent judgement.

2. BALANCED. Rule labels are distributed 134/90/45/16. Sampling the gold set in
   that proportion would put ~4 Delayed Support clips in it, far too few to
   estimate a per-class score. `--per-class` draws an equal number from each
   rule class instead. That biases the gold set's PRIOR, so the correct summary
   statistics on it are macro-F1 and per-class recall, never raw accuracy -
   which is what should be reported anyway given the imbalance.

The annotator's own uncertainty is recorded ('u'), not forced into a class.
A clip humans cannot label is evidence about the taxonomy, not a gap to fill.

USAGE
-----
    python fyp/annotate.py dataset/dataset --out gold_labels.csv --per-class 40

    1 / 2 / 3 / 4   assign class          SPACE  replay
    u               unclear / none fit    b      back one clip
    s               skip                  q      save and quit

Progress is saved after every keypress, so the session can be stopped and
resumed at any time; already-labelled clips are not shown again.

Then measure what the labels are for:
    python fyp/ml_classifier.py training_csv --gold gold_labels.csv
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

CLASS_NAMES = [
    "Coordinated Attack",
    "Coordinated Defense",
    "Delayed Support",
    "Spacing Breakdown",
]
UNCLEAR = "Unclear"

FOLDER_TO_LABEL = {
    "Coordinated_Attack": "Coordinated Attack",
    "Coordinated_Defense": "Coordinated Defense",
    "Delayed_Support": "Delayed Support",
    "Spacing_Breakdown": "Spacing Breakdown",
}

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


def csv_name_for(clip_path: str) -> str:
    """Map a clip video path to the training CSV filename that carries its rule
    label, so `ml_classifier --gold` can join the two."""
    folder = os.path.basename(os.path.dirname(clip_path))
    stem = os.path.splitext(os.path.basename(clip_path))[0]
    return f"{folder}__{stem}.csv"


def collect_clips(dataset_dir: str, per_class: int, seed: int = 0):
    """Balanced, shuffled sample. Returns [(clip_path, csv_name)]."""
    rng = random.Random(seed)
    chosen = []
    for folder in sorted(FOLDER_TO_LABEL):
        d = os.path.join(dataset_dir, folder)
        if not os.path.isdir(d):
            continue
        vids = [os.path.join(d, f) for f in sorted(os.listdir(d))
                if f.lower().endswith(VIDEO_EXTS)]
        rng.shuffle(vids)
        chosen.extend(vids[:per_class] if per_class > 0 else vids)

    # Shuffle ACROSS folders: consecutive same-class clips would leak the rule
    # label through ordering even though the name is hidden.
    rng.shuffle(chosen)
    return [(p, csv_name_for(p)) for p in chosen]


def load_existing(out_csv: str) -> dict:
    done = {}
    if os.path.exists(out_csv):
        with open(out_csv, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row.get("clip"):
                    done[row["clip"]] = row.get("label", "")
    return done


def save(out_csv: str, rows: dict, annotator: str) -> None:
    stamp = _dt.datetime.now().isoformat(timespec="seconds")
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["clip", "label", "annotator", "labelled_at"])
        for clip, label in rows.items():
            w.writerow([clip, label, annotator, stamp])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Blind tactical clip annotation.")
    ap.add_argument("dataset_dir", help="Clip tree with the 4 class folders.")
    ap.add_argument("--out", default="gold_labels.csv")
    ap.add_argument("--per-class", type=int, default=40,
                    help="Clips sampled per rule class (default 40 -> 160 total, "
                         "about 1.5-2 h of labelling).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--annotator", default=os.environ.get("USERNAME", "human"))
    ap.add_argument("--fps", type=int, default=15,
                    help="Playback fps. Slower than 30 helps judge timing, "
                         "which is what Delayed Support turns on.")
    args = ap.parse_args(argv)

    try:
        import cv2
    except ImportError:
        print("opencv-python is required: pip install opencv-python",
              file=sys.stderr)
        return 1

    clips = collect_clips(args.dataset_dir, args.per_class, args.seed)
    if not clips:
        print(f"No clips found under {args.dataset_dir!r}", file=sys.stderr)
        return 1

    labelled = load_existing(args.out)
    todo = [(p, n) for (p, n) in clips if n not in labelled]

    print("=" * 66)
    print("BLIND TACTICAL ANNOTATION")
    print("=" * 66)
    print(f"  sampled       : {len(clips)} clips "
          f"({args.per_class}/class, shuffled across classes)")
    print(f"  already done  : {len(labelled)}")
    print(f"  to label now  : {len(todo)}")
    print(f"  output        : {args.out}")
    print("\n  The rule engine's label is deliberately hidden. Judge each clip")
    print("  on what you see - disagreement with the rules is the measurement.\n")
    for i, c in enumerate(CLASS_NAMES, 1):
        print(f"    {i}  {c}")
    print("    u  Unclear / none fit      s  skip")
    print("    SPACE replay   b back   q save and quit\n")

    key_to_label = {ord(str(i + 1)): c for i, c in enumerate(CLASS_NAMES)}
    idx = 0
    while 0 <= idx < len(todo):
        clip_path, csv_name = todo[idx]
        cap = cv2.VideoCapture(clip_path)
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
        cap.release()
        if not frames:
            idx += 1
            continue

        delay = max(int(1000 / max(args.fps, 1)), 1)
        decision = None
        while decision is None:
            for fr in frames:
                shown = fr.copy()
                h = shown.shape[0]
                cv2.rectangle(shown, (0, h - 42), (shown.shape[1], h),
                              (0, 0, 0), -1)
                cv2.putText(shown,
                            f"[{idx + 1}/{len(todo)}]  "
                            f"1 Attack  2 Defense  3 Delayed  4 Spacing  "
                            f"u unclear  b back  q quit",
                            (8, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                            (255, 255, 255), 1)
                cv2.imshow("annotate", shown)
                k = cv2.waitKey(delay) & 0xFF
                if k == 255:
                    continue
                if k in key_to_label:
                    decision = key_to_label[k]
                elif k == ord("u"):
                    decision = UNCLEAR
                elif k == ord("s"):
                    decision = "__skip__"
                elif k == ord("b"):
                    decision = "__back__"
                elif k == ord("q"):
                    decision = "__quit__"
                elif k == ord(" "):
                    decision = "__replay__"
                if decision:
                    break
            if decision == "__replay__":
                decision = None

        if decision == "__quit__":
            break
        if decision == "__back__":
            idx = max(idx - 1, 0)
            prev = todo[idx][1]
            labelled.pop(prev, None)
            continue
        if decision != "__skip__":
            labelled[csv_name] = decision
            save(args.out, labelled, args.annotator)
        idx += 1

    cv2.destroyAllWindows()
    save(args.out, labelled, args.annotator)

    counts: dict[str, int] = {}
    for v in labelled.values():
        counts[v] = counts.get(v, 0) + 1
    print(f"\nSaved {len(labelled)} labels to {args.out}")
    for k in CLASS_NAMES + [UNCLEAR]:
        if counts.get(k):
            print(f"  {k:<22}: {counts[k]}")
    usable = sum(counts.get(c, 0) for c in CLASS_NAMES)
    print(f"\n  usable (non-Unclear): {usable}")
    if usable < 40:
        print("  Fewer than ~40 usable labels gives very wide confidence "
              "intervals;\n  keep going before quoting per-class numbers.")
    print("\nNow measure rules-vs-human and model-vs-human:")
    print(f"  python fyp/ml_classifier.py training_csv --gold {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
