"""
train_mamba.py – Training pipeline for the Mamba volleyball classifier.

Loads labelled CSV files produced by label_clips.py (training window,
frames 1–29, with a `target_label` column), trains the MambaClassifier,
and saves a checkpoint for later use by infer_mamba.py.

Usage
-----
    python train_mamba.py <csv_directory> [options]

Example
-------
    python train_mamba.py ./labelled_clips \
        --epochs 50 \
        --batch_size 32 \
        --d_model 64 \
        --n_layers 4 \
        --lr 1e-3 \
        --checkpoint mamba_checkpoint.pt
"""

from __future__ import annotations

import argparse
import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from mamba_model import (
    FEATURE_COLS,
    INPUT_DIM,
    LABEL_NAMES,
    NUM_CLASSES,
    MambaClassifier,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_STD = 1e-6  # minimum std used when normalising features


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_confusion_metrics(cm: torch.Tensor) -> dict[str, float]:
    """
    Compute accuracy, macro-F1 and balanced accuracy from a confusion matrix.

    cm layout: rows=true labels, cols=predicted labels.
    """
    total = float(cm.sum().item())
    correct = float(torch.diag(cm).sum().item())
    accuracy = (correct / total) if total > 0 else 0.0

    tp = torch.diag(cm).float()
    support = cm.sum(dim=1).float()
    pred_count = cm.sum(dim=0).float()

    precision = tp / pred_count.clamp(min=1.0)
    recall = tp / support.clamp(min=1.0)
    f1 = (2.0 * precision * recall) / (precision + recall).clamp(min=1e-12)

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(f1.mean().item()),
        "balanced_acc": float(recall.mean().item()),
    }


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def _seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class VolleyballDataset(Dataset):
    """
    Loads all labelled CSV files from a directory.

    Each CSV must have:
      • Columns: frame_id, ball_x, ball_y, p1_x, p1_y, …, p6_x, p6_y,
                 target_label
      • Rows: exactly the training window (frames 1–29, i.e. ≤ 29 rows)

    Returns
    -------
    (seq_tensor, label_idx)
      seq_tensor : (29, 14) float32
      label_idx  : int64 scalar
    """

    def __init__(self, directory: str) -> None:
        self.samples: list[tuple[torch.Tensor, int]] = []
        self.label_to_idx = {name: i for i, name in enumerate(LABEL_NAMES)}

        csv_files = [f for f in os.listdir(directory) if f.lower().endswith(".csv")]
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in '{directory}'.")

        skipped = 0
        for fname in csv_files:
            fpath = os.path.join(directory, fname)
            try:
                df = pd.read_csv(fpath)
            except (pd.errors.ParserError, OSError, ValueError) as exc:
                print(f"[SKIP] {fname}: {exc}")
                skipped += 1
                continue

            if "target_label" not in df.columns:
                print(f"[SKIP] {fname}: missing 'target_label' column.")
                skipped += 1
                continue

            missing_feat = [c for c in FEATURE_COLS if c not in df.columns]
            if missing_feat:
                print(f"[SKIP] {fname}: missing feature columns {missing_feat}.")
                skipped += 1
                continue

            label_str = df["target_label"].iloc[0]
            if label_str not in self.label_to_idx:
                print(f"[SKIP] {fname}: unknown label '{label_str}'.")
                skipped += 1
                continue

            label_idx = self.label_to_idx[label_str]

            features = df[FEATURE_COLS].values.astype(np.float32)  # (T, 14)
            # Pad or truncate to exactly 29 frames
            target_len = 29
            if len(features) < target_len:
                pad = np.zeros((target_len - len(features), INPUT_DIM), dtype=np.float32)
                features = np.vstack([features, pad])
            else:
                features = features[:target_len]

            seq_tensor = torch.from_numpy(features)                  # (29, 14)
            self.samples.append((seq_tensor, label_idx))

        print(
            f"Loaded {len(self.samples)} samples from '{directory}' "
            f"({skipped} skipped)."
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.samples[idx]


# ---------------------------------------------------------------------------
# Normalisation statistics (computed from training split)
# ---------------------------------------------------------------------------

def compute_normalization(
    dataset: VolleyballDataset,
    train_indices: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-feature mean and std computed over training sequences."""
    seqs = torch.stack([dataset.samples[i][0] for i in train_indices])  # (N, 29, 14)
    mean = seqs.mean(dim=(0, 1))   # (14,)
    std = seqs.std(dim=(0, 1)).clamp(min=MIN_STD)
    return mean, std


def compute_class_weights(
    dataset: VolleyballDataset,
    train_indices: list[int],
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from the training split.

    Weights are normalised so their mean equals 1, which preserves the
    effective learning rate while up-weighting minority classes.
    """
    counts = torch.zeros(num_classes)
    for i in train_indices:
        _, label_idx = dataset.samples[i]
        counts[label_idx] += 1
    # Inverse frequency; +1 avoids division-by-zero for unseen classes
    weights = num_classes / (counts + 1.0)
    weights = weights / weights.mean()
    return weights.to(device)


def stratified_split_indices(
    dataset: VolleyballDataset,
    val_split: float,
    test_split: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    """Build stratified train/val/test index splits based on class labels."""
    rng = random.Random(seed)

    by_label: dict[int, list[int]] = defaultdict(list)
    for i, (_, label_idx) in enumerate(dataset.samples):
        by_label[int(label_idx)].append(i)

    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    for label_idx, indices in by_label.items():
        rng.shuffle(indices)
        n = len(indices)

        n_test = int(np.floor(n * test_split))
        n_val = int(np.floor(n * val_split))

        if n >= 3 and test_split > 0 and n_test == 0:
            n_test = 1
        if n >= 3 and val_split > 0 and n_val == 0:
            n_val = 1

        while n - n_val - n_test < 1 and (n_val > 0 or n_test > 0):
            if n_val >= n_test and n_val > 0:
                n_val -= 1
            elif n_test > 0:
                n_test -= 1

        test_part = indices[:n_test]
        val_part = indices[n_test:n_test + n_val]
        train_part = indices[n_test + n_val:]

        train_indices.extend(train_part)
        val_indices.extend(val_part)
        test_indices.extend(test_part)

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)
    return train_indices, val_indices, test_indices


def label_distribution(dataset: VolleyballDataset, indices: list[int]) -> dict[str, int]:
    """Return split label counts keyed by label name."""
    counts = {name: 0 for name in LABEL_NAMES}
    for i in indices:
        _, label_idx = dataset.samples[i]
        counts[LABEL_NAMES[int(label_idx)]] += 1
    return counts


class NormWrapper(Dataset):
    """Wraps a dataset and applies per-feature z-score normalisation."""

    def __init__(
        self,
        base: VolleyballDataset,
        indices: list[int],
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> None:
        self.base = base
        self.indices = indices
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        seq, label = self.base.samples[self.indices[i]]
        return (seq - self.mean) / self.std, label


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------

def _run_epoch(
    model: MambaClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    """Run one epoch; return (avg_loss, accuracy)."""
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(training):
        for seqs, labels in loader:
            seqs = seqs.to(device)
            labels = labels.to(device)

            logits = model(seqs)
            loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping for stability
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += len(labels)

    return total_loss / total, correct / total


def _evaluate(
    model: MambaClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> dict[str, object]:
    """Evaluate model and return loss + classification metrics + confusion matrix."""
    model.eval()

    total_loss = 0.0
    total = 0
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    with torch.no_grad():
        for seqs, labels in loader:
            seqs = seqs.to(device)
            labels = labels.to(device)

            logits = model(seqs)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=-1)

            total_loss += loss.item() * len(labels)
            total += len(labels)

            for t, p in zip(labels.detach().cpu().tolist(), preds.detach().cpu().tolist()):
                cm[int(t), int(p)] += 1

    metrics = _compute_confusion_metrics(cm)
    return {
        "loss": (total_loss / total) if total > 0 else 0.0,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "balanced_acc": metrics["balanced_acc"],
        "confusion": cm,
    }


# ---------------------------------------------------------------------------
# Main training script
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    _seed_everything(args.seed)

    if args.val_split < 0 or args.test_split < 0 or (args.val_split + args.test_split) >= 1.0:
        print("[ERROR] val_split and test_split must be >= 0 and val_split + test_split < 1.")
        return

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────
    full_dataset = VolleyballDataset(args.csv_dir)
    if len(full_dataset) == 0:
        print("[ERROR] Dataset is empty – nothing to train on.")
        return

    n_total = len(full_dataset)
    train_indices, val_indices, test_indices = stratified_split_indices(
        full_dataset,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
    )

    n_train = len(train_indices)
    n_val = len(val_indices)
    n_test = len(test_indices)
    if n_train == 0:
        print("[ERROR] Training split is empty. Reduce val/test split ratios.")
        return

    mean, std = compute_normalization(full_dataset, train_indices)

    train_ds = NormWrapper(full_dataset, train_indices, mean, std)
    val_ds = NormWrapper(full_dataset, val_indices, mean, std) if n_val > 0 else None
    test_ds = NormWrapper(full_dataset, test_indices, mean, std) if n_test > 0 else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
    )
    val_loader = (
        DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        if val_ds is not None
        else None
    )
    test_loader = (
        DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        if test_ds is not None
        else None
    )

    print(f"Train: {n_train} | Val: {n_val} | Test: {n_test}")

    train_dist = label_distribution(full_dataset, train_indices)
    val_dist = label_distribution(full_dataset, val_indices) if n_val > 0 else {}
    test_dist = label_distribution(full_dataset, test_indices) if n_test > 0 else {}
    print("Split distribution:")
    for label in LABEL_NAMES:
        t = train_dist.get(label, 0)
        v = val_dist.get(label, 0)
        te = test_dist.get(label, 0)
        print(f"  {label:22s} train={t:4d}  val={v:4d}  test={te:4d}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = MambaClassifier(
        input_dim=INPUT_DIM,
        d_model=args.d_model,
        n_layers=args.n_layers,
        d_state=args.d_state,
        d_conv=args.d_conv,
        num_classes=NUM_CLASSES,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    # ── Optimiser & scheduler ──────────────────────────────────────────────
    class_weights = compute_class_weights(full_dataset, train_indices, NUM_CLASSES, device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # ── Training loop ──────────────────────────────────────────────────────
    best_val_macro_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = _run_epoch(model, train_loader, criterion, optimizer, device)
        if val_loader is not None:
            val_metrics = _evaluate(model, val_loader, criterion, device, NUM_CLASSES)
        else:
            val_metrics = {
                "loss": train_loss,
                "accuracy": train_acc,
                "macro_f1": train_acc,
                "balanced_acc": train_acc,
                "confusion": torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64),
            }
        scheduler.step()

        history_row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_metrics["loss"]),
            "val_acc": float(val_metrics["accuracy"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
            "val_balanced_acc": float(val_metrics["balanced_acc"]),
        }
        history.append(history_row)

        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
            f"val_loss={val_metrics['loss']:.4f}  val_acc={val_metrics['accuracy']:.3f}  "
            f"val_f1={val_metrics['macro_f1']:.3f}"
        )

        monitor = float(val_metrics["macro_f1"])
        if monitor >= best_val_macro_f1:
            best_val_macro_f1 = monitor
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "norm_mean": mean,
                    "norm_std": std,
                    "args": vars(args),
                    "label_names": LABEL_NAMES,
                    "best_val_macro_f1": best_val_macro_f1,
                },
                args.checkpoint,
            )
        else:
            patience_counter += 1
            if args.patience > 0 and patience_counter >= args.patience:
                print(
                    f"\nEarly stopping: no val_acc improvement for "
                    f"{args.patience} consecutive epochs."
                )
                break

    if args.history_csv:
        pd.DataFrame(history).to_csv(args.history_csv, index=False)
        print(f"Training history saved to '{args.history_csv}'.")

    # Final unseen-data evaluation with the best checkpoint.
    test_summary = ""
    if test_loader is not None:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        test_metrics = _evaluate(model, test_loader, criterion, device, NUM_CLASSES)

        print("\nHeld-out test metrics (unseen data):")
        print(f"  test_loss       : {test_metrics['loss']:.4f}")
        print(f"  test_acc        : {test_metrics['accuracy']:.3f}")
        print(f"  test_macro_f1   : {test_metrics['macro_f1']:.3f}")
        print(f"  test_balanced_acc: {test_metrics['balanced_acc']:.3f}")

        cm = test_metrics["confusion"]
        print("  confusion matrix (rows=true, cols=pred):")
        print(cm)

        test_summary = (
            f" Test acc={test_metrics['accuracy']:.3f}, "
            f"macro_f1={test_metrics['macro_f1']:.3f}, "
            f"balanced_acc={test_metrics['balanced_acc']:.3f}."
        )

    print(
        f"\nTraining complete. Best val_macro_f1={best_val_macro_f1:.3f} "
        f"at epoch {best_epoch}. Checkpoint saved to '{args.checkpoint}'."
        f"{test_summary}"
    )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Mamba volleyball play classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    parser.add_argument(
        "csv_dir",
        help="Directory of labelled CSV files (output of label_clips.py).",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Fraction of data held out for validation.",
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.1,
        help="Fraction of data held out for final unseen-data testing.",
    )
    # Model
    parser.add_argument("--d_model", type=int, default=64,  help="Model dimension.")
    parser.add_argument("--n_layers", type=int, default=4,  help="Number of Mamba blocks.")
    parser.add_argument("--d_state", type=int, default=16,  help="SSM state dimension.")
    parser.add_argument("--d_conv", type=int, default=4,    help="Depthwise conv kernel width.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")
    # Training
    parser.add_argument("--epochs", type=int, default=50,   help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3,   help="Initial learning rate.")
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="AdamW weight decay."
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.1,
        help="Label smoothing factor for cross-entropy.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader worker processes.",
    )
    parser.add_argument("--seed", type=int, default=42,      help="Random seed.")
    # Output
    parser.add_argument(
        "--checkpoint",
        default="mamba_checkpoint.pt",
        help="Path to save the best model checkpoint.",
    )
    parser.add_argument(
        "--history_csv",
        default="training_history.csv",
        help="Path to save per-epoch training metrics.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early-stopping patience in epochs (0 = disabled).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(_parse_args())
