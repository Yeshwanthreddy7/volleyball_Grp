"""
preflight.py - fail loudly BEFORE producing a confident-looking wrong answer.

WHY THIS EXISTS
---------------
Every serious failure in this project's history was silent. The pipeline ran to
completion, printed a formatted report, rendered a video, and was wrong:

  * §12.8  a half-court quad mapped onto the full-court plane deleted the whole
           front row; output: 100% "TRANSITION", no error.
  * §12.9  off-court coaches claimed all six identity slots; output: 100%
           "TRANSITION", no error.
  * §13    the custom detector scored 0.06 max confidence on players in the
           demo video; output: zero boxes, 100% "TRANSITION", no error.

In all three the console said "no players" while the report said "analysis
complete". A tactical report computed from zero or twelve players is not a
degraded result - it is a meaningless one, and it must not be presented as an
answer. This module samples the actual segment that is about to be processed,
measures what survives each stage, and refuses to continue when the population
is degenerate.

The verdict logic (`assess`) is deliberately separated from frame sampling and
is pure arithmetic on a stats dict, so it is unit-testable with no video, no
weights and no cv2.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Six players per side in indoor volleyball. Detections legitimately fall below
# six (occlusion behind the net band, a player leaving frame during a dig) and
# rise slightly above it during substitutions, so the ACCEPTABLE band is wide;
# only clearly impossible populations are rejected.
EXPECTED_TEAM_SIZE = 6
MIN_TEAM_MEDIAN = 3.0        # below this the six slots cannot be filled
MAX_TEAM_MEDIAN = 9.0        # above this the team split is leaking opponents
MIN_RAW_MEDIAN = 4.0         # a rally frame always shows several people
MIN_BALL_RECALL = 0.20       # below this, ball-derived features are noise

FATAL, WARN, OK = "FATAL", "WARN", "OK"


@dataclass
class Finding:
    level: str
    stage: str
    message: str
    remedy: str


@dataclass
class Verdict:
    findings: list[Finding] = field(default_factory=list)

    @property
    def fatal(self) -> bool:
        return any(f.level == FATAL for f in self.findings)

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.level == WARN]

    def render(self) -> str:
        if not self.findings:
            return "  All preflight checks passed."
        lines = []
        for f in self.findings:
            mark = "FATAL" if f.level == FATAL else "WARN "
            lines.append(f"  [{mark}] {f.stage}: {f.message}")
            lines.append(f"          -> {f.remedy}")
        return "\n".join(lines)


def assess(stats: dict) -> Verdict:
    """
    Turn measured per-stage populations into a go / no-go verdict.

    `stats` keys (all optional except the medians):
      raw_person_median      persons per frame straight from the detector
      masked_person_median   after the court mask
      team_person_median     after the team split (what the tracker feeds slots)
      ball_recall            fraction of sampled frames with a ball detection
      n_frames               how many frames were sampled
    """
    v = Verdict()
    raw = float(stats.get("raw_person_median", 0.0))
    masked = float(stats.get("masked_person_median", 0.0))
    team = float(stats.get("team_person_median", 0.0))
    ball = stats.get("ball_recall")

    # --- Stage 1: does the detector see people at all? ---------------------
    if raw < MIN_RAW_MEDIAN:
        v.findings.append(Finding(
            FATAL, "detection",
            f"median {raw:.1f} people/frame - a rally frame shows 12 players "
            f"plus officials, so the detector is not working on this footage",
            "The volleyball fine-tune collapses on unseen courts (measured "
            "0.1 players/frame on videoplayback (4)). Use stock COCO weights "
            "for players and the fine-tune only for the ball: "
            "--yolo-model yolo11n.pt --ball-model fyp/volleyball_best.pt",
        ))
        return v      # nothing downstream can be meaningful; stop here.

    # --- Stage 2: does the court mask keep them? ---------------------------
    if masked <= 0.0:
        v.findings.append(Finding(
            FATAL, "court mask",
            f"the detector found {raw:.1f} people/frame but the court mask "
            f"kept none - the calibration disagrees with the image",
            "Check the auto-detected court quad, or supply --court-corners "
            "explicitly. Run fyp/diagnose_video.py to see the quad drawn.",
        ))
        return v

    # --- Stage 3: is the analysed population one team? ---------------------
    if team <= 0.0:
        v.findings.append(Finding(
            FATAL, "team split",
            f"{masked:.1f} people/frame survive the court mask but none "
            f"survive the team split - every window will be Unclassified",
            "The geometric split fails when the court plane is mis-mapped. "
            "Use --team-split colour, and verify --team-side matches the half "
            "the analysed team actually occupies.",
        ))
    elif team < MIN_TEAM_MEDIAN:
        v.findings.append(Finding(
            FATAL, "team split",
            f"median {team:.1f} players/frame reach the six identity slots; "
            f"below {MIN_TEAM_MEDIAN:.0f} the formation features (spacing, "
            f"centroid, synchronisation) describe missing players",
            "Lower --conf-threshold, raise --imgsz, or widen the court mask; "
            "confirm the analysed team is on the --team-side you asked for.",
        ))
    elif team > MAX_TEAM_MEDIAN:
        v.findings.append(Finding(
            WARN, "team split",
            f"median {team:.1f} players/frame - more than one team's worth, so "
            f"opponents are leaking into the analysed formation",
            "Use --team-split colour (the geometric split cannot separate the "
            "two front rows at the net) and check --team-clusters covers the "
            "libero's contrasting jersey.",
        ))
    elif abs(team - EXPECTED_TEAM_SIZE) > 2.0:
        v.findings.append(Finding(
            WARN, "team split",
            f"median {team:.1f} players/frame vs the expected "
            f"{EXPECTED_TEAM_SIZE} - formation features will be noisier",
            "Usually harmless (occlusion at the net band); inspect a rendered "
            "frame if the tactical labels look implausible.",
        ))

    # --- Stage 4: is the ball usable? --------------------------------------
    if ball is not None and float(ball) < MIN_BALL_RECALL:
        v.findings.append(Finding(
            WARN, "ball",
            f"ball found in {100 * float(ball):.0f}% of sampled frames - the "
            f"ball-derived features and the Coordinated-Attack / "
            f"Delayed-Support rules are unreliable at this recall",
            "Raise --imgsz (measured recall 15% at 640 vs 77% at 1280) and "
            "pass --ball-model fyp/volleyball_best.pt; report ball-dependent "
            "results as low-confidence.",
        ))

    return v


def summarise(stats: dict) -> str:
    """One-line-per-stage table of what the preflight actually measured."""
    rows = [
        ("frames sampled", stats.get("n_frames")),
        ("people/frame (detector)", stats.get("raw_person_median")),
        ("people/frame (after court mask)", stats.get("masked_person_median")),
        ("players/frame (after team split)", stats.get("team_person_median")),
    ]
    out = []
    for label, value in rows:
        if value is None:
            continue
        out.append(f"  {label:<34}: {value:g}")
    if stats.get("ball_recall") is not None:
        out.append(f"  {'ball recall':<34}: "
                   f"{100 * float(stats['ball_recall']):.0f}%")
    return "\n".join(out)


__all__ = [
    "assess", "summarise", "Verdict", "Finding",
    "FATAL", "WARN", "OK",
    "EXPECTED_TEAM_SIZE", "MIN_TEAM_MEDIAN", "MAX_TEAM_MEDIAN",
    "MIN_RAW_MEDIAN", "MIN_BALL_RECALL",
]
