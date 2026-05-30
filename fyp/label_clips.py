"""
Automated Rule-Based Labeling Engine for Sports Analytics.

Processes a directory of CSV files containing tracked player and ball
coordinates for volleyball clips. Each CSV represents a 41-frame clip
(30 FPS) on a 1800×900 cm top-down court with the net at Y = 0.

Columns: frame_id, ball_x, ball_y, p1_x, p1_y, ..., p6_x, p6_y

Logic:
  - Evaluation Window: frames 30–41 (used to derive label)
  - Training Window:   frames  1–29 (saved to disk with target_label column)
  - "Unclassified" clips are deleted from the dataset.

Output Labels (numeric index → string)
  1 → Coordinated Attack
  2 → Coordinated Defense
  3 → Delayed Support
  4 → Spacing Breakdown

Key Parameters Used to Generate Labels
  - Player Spacing          : distance(player_i, player_j) for all pairs
  - Movement Synchronization: sync_score = mean pairwise cosine similarity
                              of player velocity vectors
  - Player Arrival Timing   : arrival_time_difference = |t_playerA − t_playerB|
  - Team Centroid Movement  : centroid = mean(player_positions) per frame
"""

import os
import sys

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# ANSI terminal color helpers
# ---------------------------------------------------------------------------

# 256-color ANSI codes that match the BGR colors used on the annotated video:
#   Orange (208) → Coordinated Attack
#   Blue   ( 12) → Coordinated Defense
#   Cyan   ( 14) → Delayed Support
#   Red    (  9) → Spacing Breakdown
_ANSI: dict[str, str] = {
    "Coordinated Attack":  "\033[38;5;208m",   # orange
    "Coordinated Defense": "\033[38;5;12m",    # blue
    "Delayed Support":     "\033[38;5;14m",    # cyan
    "Spacing Breakdown":   "\033[38;5;9m",     # red
    "reset":               "\033[0m",
}


def _c(label: str, text: str, use_color: bool) -> str:
    """Wrap *text* in the ANSI color for *label* (no-op when use_color is False)."""
    if not use_color:
        return text
    return f"{_ANSI.get(label, '')}{text}{_ANSI['reset']}"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EVAL_START = 30          # inclusive
EVAL_END = 41            # inclusive
TRAIN_END = 29           # inclusive (frames 1–29 kept)
FPS = 30                 # capture frame rate (frames per second)

PLAYER_COLS = [
    ("p1_x", "p1_y"),
    ("p2_x", "p2_y"),
    ("p3_x", "p3_y"),
    ("p4_x", "p4_y"),
    ("p5_x", "p5_y"),
    ("p6_x", "p6_y"),
]
N_PLAYERS = len(PLAYER_COLS)

# Numeric index → label string mapping (1-based, as per output specification).
# Read as: LABEL_INDEX[1] == "Coordinated Attack"
LABEL_INDEX: dict[int, str] = {
    1: "Coordinated Attack",
    2: "Coordinated Defense",
    3: "Delayed Support",
    4: "Spacing Breakdown",
}
# Reverse mapping for reporting
LABEL_TO_INDEX: dict[str, int] = {v: k for k, v in LABEL_INDEX.items()}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _player_positions(df: pd.DataFrame) -> np.ndarray:
    """
    Return shape (n_frames, 6, 2) array of player positions.

    Uses vectorised NumPy stack instead of row-by-row iteration for
    significantly faster extraction on large evaluation windows.
    """
    return np.stack(
        [df[[xc, yc]].to_numpy(dtype=float) for xc, yc in PLAYER_COLS],
        axis=1,
    )


def _nearest_neighbor_distances(positions: np.ndarray) -> np.ndarray:
    """
    Return (n_frames, 6) array of each player's minimum distance to any
    other player per frame, using vectorised broadcasting.

    Computes all C(6,2)=15 pairwise distances simultaneously per frame.
    Zero-padded `(0, 0)` slots must be filtered by the caller.
    """
    # (n_frames, 6, 1, 2) − (n_frames, 1, 6, 2) → (n_frames, 6, 6, 2)
    diff = positions[:, :, np.newaxis, :] - positions[:, np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=-1)           # (n_frames, 6, 6)
    dist[:, np.arange(N_PLAYERS), np.arange(N_PLAYERS)] = np.inf  # ignore self
    return dist.min(axis=-1)                        # (n_frames, 6)


def _compute_sync_score(positions: np.ndarray) -> float:
    """
    Compute movement synchronization score (0–1).

    Mean pairwise cosine similarity of player velocity vectors, evaluated
    per-pair across every frame transition.  Only pairs where *both* players
    are actively moving (speed ≥ 1e-6) are included, so zero-padded `(0, 0)`
    slots never corrupt the score.

    sync_score = 1  → all active players move in exactly the same direction
    sync_score = 0  → active players move perpendicularly on average
    sync_score < 0  → active players move in opposing directions on average

    Parameters
    ----------
    positions : (n_frames, 6, 2) array of player positions

    Returns
    -------
    float in [-1, 1]
    """
    if len(positions) < 2:
        return 0.0

    velocities = np.diff(positions, axis=0)  # (n_frames-1, 6, 2)
    scores: list[float] = []

    for vel_frame in velocities:
        norms = np.linalg.norm(vel_frame, axis=1)  # (6,)
        # Active players: moving and not zero-padded
        active = norms >= 1e-6
        if active.sum() < 2:
            continue
        active_vel = vel_frame[active]                          # (k, 2)
        active_norms = norms[active]                            # (k,)
        unit = active_vel / active_norms[:, np.newaxis]         # (k, 2)
        sim_matrix = unit @ unit.T                              # (k, k)
        k = int(active.sum())
        ri, ci = np.triu_indices(k, k=1)
        scores.extend(sim_matrix[ri, ci].tolist())

    return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(label_counts: dict) -> str:
    """
    Build a Team Coordination Analysis report from label counts.

    Includes per-label event counts, percentages, an ASCII bar chart, a
    coordination efficiency score, and a brief pattern-based insight.

    Each label line is printed in the same color used by the video annotation
    (orange / blue / cyan / red) when stdout is a TTY; plain text otherwise.

    Parameters
    ----------
    label_counts : mapping from label string to event count.

    Returns
    -------
    Formatted multi-line report string ready for printing.
    """
    use_color = sys.stdout.isatty()

    total = sum(label_counts.values())
    n_ca = label_counts.get("Coordinated Attack", 0)
    n_cd = label_counts.get("Coordinated Defense", 0)
    n_ds = label_counts.get("Delayed Support", 0)
    n_sb = label_counts.get("Spacing Breakdown", 0)

    coordinated = n_ca + n_cd
    efficiency = round(coordinated / total * 100) if total > 0 else 0

    def _bar(n: int, tot: int, width: int = 20) -> str:
        if tot == 0:
            return "░" * width
        filled = round(n / tot * width)
        return "█" * filled + "░" * (width - filled)

    def _pct(n: int, tot: int) -> str:
        return f"{round(n / tot * 100):3d}%" if tot > 0 else "  0%"

    sep = "=" * 56
    lines = [
        "",
        sep,
        "           Team Coordination Analysis",
        sep,
        _c("Coordinated Attack",
           f"  [1] Coordinated Attack   : {n_ca:4d}  {_pct(n_ca, total)}  {_bar(n_ca, total)}",
           use_color),
        _c("Coordinated Defense",
           f"  [2] Coordinated Defense  : {n_cd:4d}  {_pct(n_cd, total)}  {_bar(n_cd, total)}",
           use_color),
        _c("Delayed Support",
           f"  [3] Delayed Support      : {n_ds:4d}  {_pct(n_ds, total)}  {_bar(n_ds, total)}",
           use_color),
        _c("Spacing Breakdown",
           f"  [4] Spacing Breakdown    : {n_sb:4d}  {_pct(n_sb, total)}  {_bar(n_sb, total)}",
           use_color),
        "",
        f"  Total clips analysed          : {total}",
        f"  Coordination efficiency score : {efficiency}%",
    ]

    # Pattern insight
    if total > 0:
        dominant = max(label_counts, key=label_counts.get)
        insights = {
            "Spacing Breakdown":   "High Spacing Breakdown → focus on formation & positioning drills.",
            "Delayed Support":     "High Delayed Support   → improve player reaction time to ball contact.",
            "Coordinated Attack":  "High Coordinated Attack → strong offensive structure, maintain it.",
            "Coordinated Defense": "High Coordinated Defense → excellent defensive cohesion.",
        }
        qual = (
            "GOOD (≥ 60% positive)"     if efficiency >= 60 else
            "MODERATE (40–59% positive)" if efficiency >= 40 else
            "NEEDS IMPROVEMENT (< 40%)"
        )
        lines += [
            "",
            "  Insights:",
            f"  • {insights.get(dominant, '')}",
            f"  • Overall coordination: {qual}.",
        ]

    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Rule implementations
# ---------------------------------------------------------------------------

def rule1_spacing_breakdown(eval_df: pd.DataFrame) -> bool:
    """
    RULE 1 – Spacing Breakdown (Structural Failure)

    Evaluates all 15 pairwise Euclidean distances between the 6 players
    for every frame in the outcome window.

    Triggers (returns True immediately) when ANY frame satisfies:
      • max nearest-neighbour distance > 600 cm  (open hole in formation), OR
      • min nearest-neighbour distance < 50 cm   (collision / overlap).

    The threshold was raised from 400 cm to 600 cm because in normal
    volleyball play (e.g. front-row attackers and back-row setters spread
    apart during a set), nearest-neighbour distances of 400–550 cm are
    routine and should NOT be labelled as structural failures.  A gap of
    > 600 cm represents a genuinely uncovered zone on a 1800×900 cm court.

    Zero-padded `(0, 0)` player slots are excluded so that untracked
    positions never generate phantom collisions or inflated gaps.
    """
    positions = _player_positions(eval_df)  # (n_frames, 6, 2)

    for pos in positions:
        # Only consider real tracking slots
        occupied = pos[~np.all(pos == 0, axis=1)]  # (k, 2)
        if len(occupied) < 2:
            continue

        # Vectorised pairwise distances – all C(k,2) pairs at once
        diff = occupied[:, np.newaxis, :] - occupied[np.newaxis, :, :]  # (k, k, 2)
        dists = np.linalg.norm(diff, axis=-1)                           # (k, k)
        np.fill_diagonal(dists, np.inf)
        nn_dists = dists.min(axis=-1)  # (k,) nearest-neighbour per player

        if nn_dists.max() > 600 or nn_dists.min() < 50:
            return True   # early exit on first offending frame

    return False


def rule2_delayed_support(eval_df: pd.DataFrame) -> bool:
    """
    RULE 2 – Delayed Support (Temporal Failure)

    Finds the impact_frame where the ball's Y-velocity sharply changes (sign
    reversal or zero-crossing).  Identifies the player closest to the ball at
    that frame, then checks whether that player reaches their peak frame-to-frame
    speed more than 5 frames after the impact_frame.

    Falls back to detecting a sudden deceleration in combined XY ball speed when
    no clear Y-reversal is present (e.g. nearly horizontal ball trajectory).

    If the ball is never detected (all coordinates are the (0, 0) sentinel),
    the rule cannot fire because there is no meaningful impact frame.
    """
    ball_x = eval_df["ball_x"].values.astype(float)
    ball_y = eval_df["ball_y"].values.astype(float)

    # No ball detected at all – cannot determine impact frame.
    if np.all(ball_x == 0) and np.all(ball_y == 0):
        return False

    ball_y_vel = np.diff(ball_y)  # (n-1,)

    # Primary: Y-velocity sign reversal (ball bounce / contact)
    impact_rel = None
    for i in range(1, len(ball_y_vel)):
        if ball_y_vel[i - 1] * ball_y_vel[i] <= 0:
            impact_rel = i
            break

    # Fallback: sudden deceleration in combined XY speed (> 60% drop in 1 frame)
    if impact_rel is None:
        ball_speed = np.sqrt(np.diff(ball_x) ** 2 + ball_y_vel ** 2)
        for i in range(1, len(ball_speed)):
            if ball_speed[i - 1] > 5.0 and ball_speed[i] < ball_speed[i - 1] * 0.4:
                impact_rel = i
                break

    if impact_rel is None:
        return False

    impact_frame = impact_rel  # 0-based index into eval_df rows

    positions = _player_positions(eval_df)  # (n_frames, 6, 2)
    ball_pos_at_impact = np.array(
        [eval_df["ball_x"].iloc[impact_frame], eval_df["ball_y"].iloc[impact_frame]],
        dtype=float,
    )

    # Closest player to ball at impact_frame (vectorised)
    diffs = positions[impact_frame] - ball_pos_at_impact  # (6, 2)
    player_dists = np.linalg.norm(diffs, axis=-1)         # (6,)
    closest_player = int(np.argmin(player_dists))

    # Frame-to-frame speed of the closest player
    player_traj = positions[:, closest_player, :]  # (n_frames, 2)
    player_speed = np.linalg.norm(np.diff(player_traj, axis=0), axis=1)  # (n_frames-1,)

    if len(player_speed) == 0:
        return False

    reaction_frame = int(np.argmax(player_speed))  # index of peak speed

    return (reaction_frame - impact_frame) > 5


def rule3_coordinated_attack(eval_df: pd.DataFrame) -> bool:
    """
    RULE 3 – Coordinated Attack (Offensive Structure)

    Triggers when ALL three conditions hold:
      • Ball is detected and in an attacking zone (ball_y < 500 cm) in at least
        one frame of the outcome window, AND
      • Average speed of the top-2 fastest players > 300 cm/s, AND
      • Average speed of the bottom-4 players < 150 cm/s.

    Speeds are computed as frame-to-frame Euclidean displacement (cm/frame)
    and converted to cm/s by multiplying by FPS (30).

    The ball_y < 500 threshold catches attacks where the ball is on either side
    of the net (Y ≈ 450 cm) rather than restricting to a narrow baseline zone.
    The bottom-4 speed is raised to 150 cm/s because setters and support
    players legitimately adjust their position during an offensive play.
    """
    ball_x = eval_df["ball_x"].values.astype(float)
    ball_y = eval_df["ball_y"].values.astype(float)

    # (0, 0) is the sentinel used by the pipeline when the ball is undetected;
    # skip the rule entirely if the ball was never observed.
    if np.all(ball_x == 0) and np.all(ball_y == 0):
        return False

    if not (ball_y < 500).any():
        return False

    positions = _player_positions(eval_df)  # (n_frames, 6, 2)

    # Per-player average speed (cm/frame), then convert to cm/s
    speeds_cmf = np.linalg.norm(
        np.diff(positions, axis=0), axis=-1
    ).mean(axis=0)                              # (6,) – avg cm/frame per player
    speeds_cms = speeds_cmf * FPS              # cm/s

    # np.partition is O(n) vs O(n log n) sort; fine for only 6 players.
    partitioned = np.partition(speeds_cms, -2)  # top-2 in last 2 positions
    top2_avg  = partitioned[-2:].mean()
    bottom4_avg = partitioned[:-2].mean()

    return top2_avg > 300 and bottom4_avg < 150


def rule4_coordinated_defense(eval_df: pd.DataFrame) -> bool:
    """
    RULE 4 – Coordinated Defense (Defensive Structure)

    Triggers when BOTH conditions hold:
      • Team centroid velocity > 3 cm/frame (team is actively shifting
        together as a block), AND
      • std(Spatial Variance) < 30% of mean(Spatial Variance)
        (formation is maintained – players are not scattering relative to
        each other, though the formation may shift laterally).

    The centroid threshold was lowered from 10 to 3 cm/frame because
    real defensive shifts typically register at 5–9 cm/frame and the
    previous value was suppressing valid detections.

    The spatial-variance std ratio was raised from 5% to 30% to allow for
    minor positional adjustments that accompany a defensive block shift.

    Zero-padded `(0, 0)` player slots (untracked positions) are excluded from
    the centroid and spatial-variance computations to prevent phantom drift.
    """
    positions = _player_positions(eval_df)  # (n_frames, 6, 2)

    # Centroid per frame using only occupied (non-zero-padded) slots
    centroid = np.zeros((len(positions), 2), dtype=float)
    for f, pos in enumerate(positions):
        occupied = pos[~np.all(pos == 0, axis=1)]
        centroid[f] = occupied.mean(axis=0) if len(occupied) > 0 else pos.mean(axis=0)

    # Centroid velocity (frame-to-frame displacement magnitude, cm/frame)
    centroid_vel = np.linalg.norm(np.diff(centroid, axis=0), axis=1)
    if centroid_vel.mean() <= 3:
        return False

    # Spatial variance per frame: mean squared distance of occupied players to centroid
    spatial_var = np.zeros(len(positions))
    for f, pos in enumerate(positions):
        occupied = pos[~np.all(pos == 0, axis=1)]
        if len(occupied) == 0:
            continue
        diffs = occupied - centroid[f]
        spatial_var[f] = (diffs ** 2).sum(axis=1).mean()

    mean_sv = spatial_var.mean()
    std_sv  = spatial_var.std()

    if mean_sv == 0:
        return False

    return std_sv < (0.30 * mean_sv)


# ---------------------------------------------------------------------------
# Main labeling function
# ---------------------------------------------------------------------------

def label_clip(eval_df: pd.DataFrame) -> str:
    """
    Evaluate the outcome window (frames 30–41) against the 4 heuristic rules
    and return the assigned label string.

    Rules are evaluated in priority order: positive tactical patterns (Attack,
    Defense) are checked first so they are never masked by the structural
    catch-all. Spacing Breakdown is checked last and fires only when no
    organised pattern is detected.

      1. Coordinated Attack  (offensive structure)
      2. Coordinated Defense (defensive structure)
      3. Delayed Support     (temporal failure)
      4. Spacing Breakdown   (structural failure – catch-all)

    The first rule that fires determines the label; remaining rules are not
    evaluated (short-circuit evaluation).
    """
    if rule3_coordinated_attack(eval_df):
        return "Coordinated Attack"

    if rule4_coordinated_defense(eval_df):
        return "Coordinated Defense"

    if rule2_delayed_support(eval_df):
        return "Delayed Support"

    if rule1_spacing_breakdown(eval_df):
        return "Spacing Breakdown"

    return "Unclassified"


# ---------------------------------------------------------------------------
# Batch-processing loop
# ---------------------------------------------------------------------------

def process_directory(input_dir: str) -> None:
    """
    Batch-process every CSV in *input_dir*.

    For each file:
      1. Load the CSV.
      2. Extract the evaluation window (frames 30–41) and determine the label.
      3. If label is "Unclassified", delete the file and continue.
      4. Otherwise, keep only the training window (frames 1–29), append
         a `target_label` column, and overwrite the original file.

    At the end, prints a Team Coordination Analysis report with per-label
    event counts and a coordination efficiency score.
    """
    csv_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".csv")]

    if not csv_files:
        print(f"No CSV files found in '{input_dir}'.")
        return

    processed = 0
    deleted = 0
    errors = 0
    label_counts: dict[str, int] = {}

    for filename in csv_files:
        filepath = os.path.join(input_dir, filename)
        try:
            df = pd.read_csv(filepath)

            # Validate expected columns
            required_cols = {"frame_id", "ball_x", "ball_y"}
            for xc, yc in PLAYER_COLS:
                required_cols.update({xc, yc})
            missing = required_cols - set(df.columns)
            if missing:
                print(f"[SKIP]  {filename} – missing columns: {missing}")
                errors += 1
                continue

            # Split into windows (frame_id is 1-based)
            eval_df = df[(df["frame_id"] >= EVAL_START) & (df["frame_id"] <= EVAL_END)].copy()
            train_df = df[(df["frame_id"] >= 1) & (df["frame_id"] <= TRAIN_END)].copy()

            if eval_df.empty:
                print(f"[SKIP]  {filename} – evaluation window (frames 30–41) is empty.")
                errors += 1
                continue

            if train_df.empty:
                print(f"[SKIP]  {filename} – training window (frames 1–29) is empty.")
                errors += 1
                continue

            # Determine label from evaluation window
            label = label_clip(eval_df)

            if label == "Unclassified":
                os.remove(filepath)
                print(f"[DELETE] {filename} – Unclassified")
                deleted += 1
                continue

            # Append label to training window and overwrite file
            train_df = train_df.reset_index(drop=True)
            train_df["target_label"] = label
            train_df.to_csv(filepath, index=False)

            numeric_idx = LABEL_TO_INDEX.get(label, 0)
            print(f"[OK]    {filename} – [{numeric_idx}] {label}")
            label_counts[label] = label_counts.get(label, 0) + 1
            processed += 1

        except (pd.errors.ParserError, KeyError, ValueError, OSError) as exc:
            print(f"[ERROR] {filename} – {exc}")
            errors += 1

    print(
        f"\nDone. Processed: {processed}, Deleted (Unclassified): {deleted}, "
        f"Errors/Skipped: {errors}"
    )

    if label_counts:
        print(generate_report(label_counts))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python label_clips.py <directory_of_csvs>")
        sys.exit(1)

    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print(f"Error: '{directory}' is not a valid directory.")
        sys.exit(1)

    process_directory(directory)
