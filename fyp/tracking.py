"""
tracking.py – Tracking paradigm switch + Kalman tuning (spec §2A/§2B).

Two interchangeable tracker backends behind one adapter interface:

  tracker_type="bytetrack"  (default)
      IoU-only association (supervision.ByteTrack).  Fast, no appearance
      model.  Fails when two players fully overlap: after the crossing the
      IoU-only cost matrix cannot tell who is who -> ID switch.

  tracker_type="botsort"
      BoT-SORT (ultralytics implementation): IoU + optional appearance
      Re-ID + built-in GMC camera-motion compensation (sparse optical flow),
      which keeps the Kalman prediction valid while the broadcast camera
      pans/zooms.  Use for dense crossings (blocking huddles, rotations).

KALMAN TUNING FOR EXPLOSIVE KINETICS (§2B)
------------------------------------------
Both backends use the standard ByteTrack constant-velocity Kalman filter with
process noise Q and measurement noise R derived from two class attributes:

    _std_weight_position (1/20), _std_weight_velocity (1/160)

A volleyball approach-jump violates the constant-velocity assumption (~9 m/s²
horizontal acceleration bursts), so the default filter over-trusts its own
prediction and lags behind the player, exactly when tracking matters most.
`q_scale` multiplies Q (std scaled by sqrt(q_scale) inside predict only, so R
is untouched): the filter weights fresh YOLO detections more heavily than its
linear extrapolation.  q_scale = 4 raises the Kalman gain toward measurements
without making the track jitter-limited; values 2–8 are sensible.

Everything degrades gracefully: if an internals patch or an optional backend
is unavailable the adapter logs a loud warning and continues with defaults –
the pipeline never crashes because of a tracker upgrade.
"""

from __future__ import annotations

import math
import sys
from abc import ABC, abstractmethod
from types import SimpleNamespace

import numpy as np

from interfaces import DetectionResult


# ---------------------------------------------------------------------------
# Adapter interface
# ---------------------------------------------------------------------------

class BaseTrackerAdapter(ABC):
    """Uniform per-frame tracking interface: detections in, (boxes, ids) out."""

    @abstractmethod
    def update(
        self, det: DetectionResult, frame: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Advance the tracker one frame.

        Returns
        -------
        xyxy : (M, 4) tracked boxes
        ids  : (M,)   persistent integer track ids
        """

    @property
    @abstractmethod
    def name(self) -> str: ...


# ---------------------------------------------------------------------------
# Kalman patch helper (shared by both backends)
# ---------------------------------------------------------------------------

def _make_tuned_kf(base_cls, q_scale: float):
    """Subclass a ByteTrack-style KalmanFilter so predict()/multi_predict()
    scale process noise Q by `q_scale` while measurement noise R (used in
    project()/update()) keeps the default weights."""
    s = math.sqrt(max(q_scale, 1e-6))

    class TunedKF(base_cls):  # type: ignore[misc, valid-type]
        def _scaled(self, fn, *args):
            op, ov = self._std_weight_position, self._std_weight_velocity
            self._std_weight_position, self._std_weight_velocity = op * s, ov * s
            try:
                return fn(*args)
            finally:
                self._std_weight_position, self._std_weight_velocity = op, ov

        def predict(self, mean, covariance):
            return self._scaled(super().predict, mean, covariance)

        def multi_predict(self, mean, covariance):
            return self._scaled(super().multi_predict, mean, covariance)

    TunedKF.__name__ = f"TunedKF_q{q_scale:g}"
    return TunedKF


def _patch_kalman(strack_cls, tracker_obj, q_scale: float, label: str) -> bool:
    """Install a tuned Kalman filter on a ByteTrack-style tracker.  Returns
    True on success; on any internals mismatch, warns and returns False.

    IDEMPOTENT by design: `shared_kalman` is a CLASS-level (process-global)
    attribute, and a fresh tracker is constructed for every clip. The old
    implementation re-subclassed `type(shared_kalman)` on every call, so each
    clip stacked one more wrapper layer on the global filter - compounding Q
    by 4^N, exploding the covariance (overflow warnings) and finally dying
    with RecursionError after enough clips. Now the tuned class is tagged;
    re-patching with the same q_scale is a no-op, and a different q_scale
    rebuilds from the ORIGINAL base class, never from a tuned one."""
    if q_scale == 1.0:
        return True
    try:
        cur_cls = type(strack_cls.shared_kalman)
        if getattr(cur_cls, "_fyp_q_scale", None) == q_scale:
            # Already tuned globally - just mirror onto this tracker instance.
            if hasattr(tracker_obj, "kalman_filter"):
                tracker_obj.kalman_filter = cur_cls()
            return True
        base_kf = getattr(cur_cls, "_fyp_base_cls", cur_cls)
        tuned_cls = _make_tuned_kf(base_kf, q_scale)
        tuned_cls._fyp_q_scale = q_scale
        tuned_cls._fyp_base_cls = base_kf
        strack_cls.shared_kalman = tuned_cls()
        if hasattr(tracker_obj, "kalman_filter"):
            tracker_obj.kalman_filter = tuned_cls()
        # reset_id-style class attr is untouched; existing tracks keep working.
        print(f"[tracking] {label}: Kalman process noise Q scaled x{q_scale:g} "
              f"(measurement noise R unchanged).")
        return True
    except Exception as exc:  # pragma: no cover - depends on lib version
        print(f"[WARNING] {label}: could not tune Kalman filter ({exc}); "
              "using library defaults.", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# ByteTrack backend (supervision)
# ---------------------------------------------------------------------------

class ByteTrackAdapter(BaseTrackerAdapter):
    """IoU-only ByteTrack via the `supervision` package."""

    def __init__(self, fps: float = 30.0, q_scale: float = 4.0) -> None:
        import supervision as sv

        self._sv = sv
        try:
            self._tracker = sv.ByteTrack(frame_rate=int(round(fps)))
        except TypeError:  # older supervision signature
            self._tracker = sv.ByteTrack()

        # Locate the STrack class for the Kalman patch (path moved across
        # supervision versions – search defensively).
        strack_cls = None
        for path in (
            "supervision.tracker.byte_tracker.core",
            "supervision.tracker.byte_tracker.single_object_track",
        ):
            try:
                mod = __import__(path, fromlist=["STrack"])
                strack_cls = getattr(mod, "STrack", None)
                if strack_cls is not None and hasattr(strack_cls, "shared_kalman"):
                    break
                strack_cls = None
            except Exception:
                continue
        if strack_cls is not None:
            _patch_kalman(strack_cls, self._tracker, q_scale, "ByteTrack")
        elif q_scale != 1.0:
            print("[WARNING] ByteTrack: STrack internals not found; Kalman "
                  "tuning skipped.", file=sys.stderr)

    @property
    def name(self) -> str:
        return "bytetrack (IoU-only, supervision)"

    def update(
        self, det: DetectionResult, frame: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        sv = self._sv
        if len(det.person_xyxy) == 0:
            tracks = self._tracker.update_with_detections(sv.Detections.empty())
        else:
            detections = sv.Detections(
                xyxy=det.person_xyxy,
                confidence=det.person_conf,
                class_id=np.zeros(len(det.person_xyxy), dtype=int),
            )
            tracks = self._tracker.update_with_detections(detections)

        if tracks.tracker_id is None or len(tracks.xyxy) == 0:
            return np.empty((0, 4)), np.empty((0,), dtype=int)
        return tracks.xyxy, tracks.tracker_id.astype(int)


# ---------------------------------------------------------------------------
# Detection shim for ultralytics trackers
# ---------------------------------------------------------------------------

class _BoxesShim:
    """Minimal stand-in for ultralytics `Boxes` satisfying BOTH tracker APIs.

    Older BYTETracker/BOTSORT read attributes (.conf/.xywh/.xyxy/.cls); newer
    versions ALSO subscript the results object (results[mask] in
    _split_detections). SimpleNamespace only met the first contract - this
    class meets both, so the adapter works across ultralytics versions.
    """

    __slots__ = ("conf", "xywh", "xyxy", "cls")

    def __init__(self, conf, xywh, xyxy, cls):
        self.conf = np.asarray(conf, dtype=float).reshape(-1)
        self.xywh = np.asarray(xywh, dtype=float).reshape(-1, 4)
        self.xyxy = np.asarray(xyxy, dtype=float).reshape(-1, 4)
        self.cls = np.asarray(cls, dtype=float).reshape(-1)

    def __len__(self):
        return int(self.conf.shape[0])

    def __getitem__(self, idx):
        return _BoxesShim(self.conf[idx], self.xywh[idx],
                          self.xyxy[idx], self.cls[idx])

    @classmethod
    def empty(cls):
        z = np.empty((0,))
        return cls(z, np.empty((0, 4)), np.empty((0, 4)), z)

    @classmethod
    def from_xyxy(cls, xyxy, conf):
        xyxy = np.asarray(xyxy, dtype=float).reshape(-1, 4)
        if len(xyxy):
            xywh = np.stack([
                (xyxy[:, 0] + xyxy[:, 2]) / 2.0,
                (xyxy[:, 1] + xyxy[:, 3]) / 2.0,
                xyxy[:, 2] - xyxy[:, 0],
                xyxy[:, 3] - xyxy[:, 1],
            ], axis=1)
        else:
            xywh = np.empty((0, 4))
        return cls(np.asarray(conf, dtype=float), xywh, xyxy,
                   np.zeros(len(xyxy), dtype=float))


# ---------------------------------------------------------------------------
# BoT-SORT backend (ultralytics) – appearance + motion + GMC
# ---------------------------------------------------------------------------

class BoTSORTAdapter(BaseTrackerAdapter):
    """
    BoT-SORT from the ultralytics package, fed with OUR detections (so the
    detector stays behind the BaseDetector interface and is not re-run).

    * gmc_method="sparseOptFlow": global camera-motion compensation – track
      predictions are warped by the estimated inter-frame camera motion, so a
      panning broadcast camera does not shear every Kalman prediction.
    * with_reid=True adds an appearance embedding to the association cost, so
      identity survives a full IoU dropout during blocking huddles.  It costs
      extra compute; enable it when ID switches are the bottleneck.
    """

    def __init__(
        self,
        fps: float = 30.0,
        q_scale: float = 4.0,
        with_reid: bool = False,
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        proximity_thresh: float = 0.5,
        appearance_thresh: float = 0.25,
    ) -> None:
        try:
            from ultralytics.trackers.bot_sort import BOTSORT, BOTrack
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "ultralytics is required for BoT-SORT (pip install ultralytics)"
            ) from exc

        args = SimpleNamespace(
            tracker_type="botsort",
            track_high_thresh=track_high_thresh,
            track_low_thresh=track_low_thresh,
            new_track_thresh=new_track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            fuse_score=True,
            gmc_method="sparseOptFlow",
            proximity_thresh=proximity_thresh,
            appearance_thresh=appearance_thresh,
            with_reid=with_reid,
            model="auto",
        )
        self._with_reid = with_reid
        import inspect
        _kw = {}
        if "frame_rate" in inspect.signature(BOTSORT.__init__).parameters:
            _kw["frame_rate"] = int(round(fps))   # older ultralytics API

        def _build():
            t = BOTSORT(args, **_kw)              # newer API: no frame_rate
            _patch_kalman(BOTrack, t, q_scale, "BoT-SORT")
            return t

        # Fail-fast self-test: push one synthetic frame through the REAL
        # installed tracker. Any internal-API drift in ultralytics surfaces
        # HERE, so create_tracker() falls back to ByteTrack loudly instead of
        # crashing minutes into an extraction run.
        self._tracker = _build()
        self.update(
            DetectionResult(
                person_xyxy=np.array([[4.0, 4.0, 20.0, 30.0],
                                      [30.0, 8.0, 50.0, 40.0]]),
                person_conf=np.array([0.9, 0.8]),
            ),
            np.zeros((64, 64, 3), dtype=np.uint8),
        )
        self._tracker = _build()                  # fresh state for real use

    @property
    def name(self) -> str:
        reid = "ReID on" if self._with_reid else "ReID off"
        return f"botsort (motion+appearance, GMC sparseOptFlow, {reid}, ultralytics)"

    def update(
        self, det: DetectionResult, frame: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(det.person_xyxy) == 0:
            # keep internal age/GMC state moving even with no detections
            shim = _BoxesShim.empty()
        else:
            shim = _BoxesShim.from_xyxy(det.person_xyxy, det.person_conf)

        try:
            out = self._tracker.update(shim, frame)
        except TypeError:
            # older signature without img
            out = self._tracker.update(shim)

        if out is None or len(out) == 0:
            return np.empty((0, 4)), np.empty((0,), dtype=int)
        out = np.asarray(out)
        return out[:, :4].astype(float), out[:, 4].astype(int)


# ---------------------------------------------------------------------------
# Factory (spec: single 'tracker_type' configuration flag)
# ---------------------------------------------------------------------------

def create_tracker(
    tracker_type: str = "bytetrack",
    fps: float = 30.0,
    q_scale: float = 4.0,
    with_reid: bool = False,
) -> BaseTrackerAdapter:
    """Instantiate a tracker backend by name; falls back to ByteTrack with a
    warning if the requested backend cannot be constructed."""
    tracker_type = tracker_type.lower()
    if tracker_type == "botsort":
        try:
            return BoTSORTAdapter(fps=fps, q_scale=q_scale, with_reid=with_reid)
        except Exception as exc:
            print(
                f"[WARNING] BoT-SORT unavailable ({exc}); falling back to "
                "ByteTrack.",
                file=sys.stderr,
            )
            return ByteTrackAdapter(fps=fps, q_scale=q_scale)
    if tracker_type == "bytetrack":
        return ByteTrackAdapter(fps=fps, q_scale=q_scale)
    raise ValueError(f"Unknown tracker_type '{tracker_type}' (bytetrack|botsort)")


__all__ = [
    "BaseTrackerAdapter", "ByteTrackAdapter", "BoTSORTAdapter", "create_tracker",
]
