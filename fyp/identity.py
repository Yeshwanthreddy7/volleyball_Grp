"""
identity.py - ONLINE identity-consistent slot assignment (numpy-only).

Why this exists (the "tracking id is not consistent" question)
--------------------------------------------------------------
Identity is enforced in TWO stages in this project:

  Stage 1 (online, this module): the tracker's raw ids are mapped to six
  persistent player SLOTS while the video is being processed. The previous
  implementation freed a slot the instant its track id disappeared, so after
  a block/occlusion the re-detected player often landed in a DIFFERENT slot
  (and its old slot got recycled by someone else). Column p_i then silently
  changed physical player - the annotated video showed id flicker and the raw
  CSVs carried column churn.

  Stage 2 (offline, features.py): recover_identity() re-threads the stored
  per-frame positions by Hungarian assignment (250 cm link gate) and
  interpolate_gaps() bridges <=15-frame occlusions. This runs inside
  build_model_sequence(), i.e. for training AND serving identically.

SlotManager fixes Stage 1 with three rules:
  R1  a tracker id keeps its slot for life (never re-sorted);
  R2  when an id disappears, its slot is held in LIMBO for up to `max_gap`
      frames (spec 2C: 15 frames ~ 0.5 s) with its last court position;
      a NEW id appearing within a distance gate of a limbo slot INHERITS it
      (identity bridged across the occlusion even though the tracker issued
      a fresh id). The gate grows with gap length (players move while hidden):
      gate = min(gate_cm_per_frame * gap, gate_cap_cm).
  R3  only after `max_gap` frames is a slot truly freed; at most `n_slots`
      players are ever tracked (court mask upstream removes non-players).

Gate defaults are consistent with Stage 2: 60 cm/frame (~18 m/s ceiling, above
any human sprint) capped at 250 cm - the same cap recover_identity uses.
"""
from __future__ import annotations

import numpy as np


class SlotManager:
    """Persistent tracker-id -> slot assignment with occlusion bridging."""

    def __init__(
        self,
        n_slots: int = 6,
        max_gap: int = 15,
        gate_cm_per_frame: float = 60.0,
        gate_cap_cm: float = 250.0,
    ) -> None:
        self.n_slots = int(n_slots)
        self.max_gap = int(max_gap)
        self.gate_cm_per_frame = float(gate_cm_per_frame)
        self.gate_cap_cm = float(gate_cap_cm)
        self.id_to_slot: dict[int, int] = {}
        self.last_pos: dict[int, tuple[float, float]] = {}   # slot -> court cm
        self.gone_for: dict[int, int] = {}                   # slot -> frames unseen
        self.n_bridges = 0                                   # diagnostics

    # ------------------------------------------------------------------ #
    def assign(self, court_feet: dict[int, tuple[float, float]]) -> dict[int, int]:
        """Advance one frame. court_feet: {tracker_id: (x_cm, y_cm)}.

        Returns {tracker_id: slot} for ids that own a slot this frame.
        Deterministic for identical input streams.
        """
        seen = {
            int(t): (float(p[0]), float(p[1]))
            for t, p in court_feet.items()
            if p is not None and np.all(np.isfinite(np.asarray(p, dtype=float)))
        }

        # R1 - live continuation: an id we already know keeps its slot.
        live: dict[int, int] = {}                            # slot -> tid
        for tid, s in self.id_to_slot.items():
            if tid in seen:
                live[s] = tid

        # R2/R3 - age limbo slots; expire after max_gap.
        for s in list(self.last_pos):
            if s in live:
                self.gone_for[s] = 0
            else:
                self.gone_for[s] = self.gone_for.get(s, 0) + 1
                if self.gone_for[s] > self.max_gap:
                    self.last_pos.pop(s, None)
                    self.gone_for.pop(s, None)
                    for t, ss in list(self.id_to_slot.items()):
                        if ss == s:
                            del self.id_to_slot[t]

        # R2 - bridge new ids onto limbo slots, globally nearest-first.
        new_ids = sorted(t for t in seen if t not in self.id_to_slot)
        limbo = [s for s in self.last_pos if s not in live]
        pairs: list[tuple[float, int, int]] = []
        for t in new_ids:
            p = np.asarray(seen[t], dtype=float)
            for s in limbo:
                d = float(np.hypot(*(p - np.asarray(self.last_pos[s], dtype=float))))
                gap = max(1, self.gone_for.get(s, 1))
                gate = min(self.gate_cm_per_frame * gap, self.gate_cap_cm)
                if d <= gate:
                    pairs.append((d, t, s))
        pairs.sort()
        taken_t: set[int] = set()
        taken_s: set[int] = set()
        for d, t, s in pairs:
            if t in taken_t or s in taken_s:
                continue
            for old, ss in list(self.id_to_slot.items()):
                if ss == s:
                    del self.id_to_slot[old]
            self.id_to_slot[t] = s
            live[s] = t
            taken_t.add(t)
            taken_s.add(s)
            self.n_bridges += 1

        # R3 - remaining new ids get genuinely free slots (never limbo ones).
        for t in sorted(set(new_ids) - taken_t):
            reserved = set(live) | set(self.last_pos)
            free = next((k for k in range(self.n_slots) if k not in reserved), None)
            if free is None:
                continue                                     # court already full
            self.id_to_slot[t] = free
            live[free] = t

        # Book-keeping for live slots.
        for s, t in live.items():
            self.last_pos[s] = seen[t]
            self.gone_for[s] = 0

        return dict(self.id_to_slot)


__all__ = ["SlotManager"]
