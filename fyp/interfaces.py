"""
interfaces.py – Model-agnostic abstraction layer (spec §5).

Two hard boundaries decouple the pipeline from any specific model family:

  1. DETECTION INTERFACE (§5A)
     `BaseDetector.predict(frame) -> DetectionResult`.  The backend switches
     between YOLOv8 / YOLOv9 / YOLOv11 (or any ultralytics-compatible custom
     model) by changing ONE initialisation string; downstream tracking and
     feature engineering never see the difference.

  2. TEMPORAL MODEL INTERFACE (§5B)
     `TemporalClassifier.classify(seq) -> ClassificationResult` for any
     sequence model consuming a (seq_len, feature_dim) tensor.  The concrete
     architecture (pure-PyTorch Mamba SSM, Transformer baseline, future
     Bi-Mamba) is chosen from the checkpoint's own metadata, so swapping or
     benchmarking architectures never touches file I/O or evaluation loops.

Both interfaces also centralise the safety checks (feature-version guard,
Shannon-entropy computation, low-confidence anomaly flag) so every entry point
(pipeline.py, infer_mamba.py) behaves identically.
"""

from __future__ import annotations

import math
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from detect_utils import resolve_class_ids
from features import FEATURE_VERSION, MODEL_INPUT_DIM


# ---------------------------------------------------------------------------
# 1. Detection interface
# ---------------------------------------------------------------------------

# Inference resolution.  Measured on this project's own footage (audit in
# Technical_Review_and_QA.md §13): a volleyball occupies ~15-20 px on a 1280x720
# broadcast frame, so ultralytics' 640 default downscales it to ~9 px - below
# the smallest anchor stride.  Ball recall on 300 consecutive rally frames of
# "videoplayback (4)": 15% at imgsz=640 vs 77% at imgsz=1280 with the SAME
# weights.  Player recall improves too (stock yolo11n: 19.5 -> 23.3 per frame).
# 1280 is therefore the default; 640 remains available for a fast debug pass.
DEFAULT_IMGSZ = 1280


@dataclass
class DetectionResult:
    """Backend-agnostic single-frame detection output."""
    person_xyxy: np.ndarray                    # (N, 4) float
    person_conf: np.ndarray                    # (N,)   float
    ball_center: np.ndarray | None = None      # (2,) pixel centre or None
    keypoints: np.ndarray | None = None        # (N, 17, 3) x,y,conf or None

    @staticmethod
    def empty() -> "DetectionResult":
        return DetectionResult(
            person_xyxy=np.empty((0, 4), dtype=float),
            person_conf=np.empty((0,), dtype=float),
        )


class BaseDetector(ABC):
    """Abstract player/ball detector.  Implementations must be stateless
    per-frame: everything temporal (tracking, smoothing) lives downstream."""

    @abstractmethod
    def predict(self, frame: np.ndarray) -> DetectionResult:
        """Run detection on a single BGR frame."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend identifier (for logs / reports)."""


class UltralyticsDetector(BaseDetector):
    """
    Detector backed by any ultralytics-loadable weights file:
    'yolov8n.pt', 'yolov9c.pt', 'yolo11n.pt', a fine-tuned 'best.pt', …

    Class ids are resolved BY NAME from the model's own metadata
    (detect_utils.resolve_class_ids), so COCO weights (person=0, ball=32) and
    custom volleyball weights (e.g. Roboflow: ball=0, player=1) both work
    without code changes.
    """

    def __init__(
        self,
        weights: str = "yolo11n.pt",
        conf_threshold: float = 0.35,
        person_class_id: int | None = None,
        ball_class_id: int | None = None,
        device: str | None = None,
        imgsz: int = DEFAULT_IMGSZ,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "ultralytics is required for UltralyticsDetector "
                "(pip install ultralytics>=8.0.0)"
            ) from exc

        self._weights = weights
        self._conf = conf_threshold
        self._imgsz = int(imgsz)
        self._model = YOLO(weights)
        if device:
            self._model.to(device)

        names = getattr(self._model, "names", None) or {0: "person"}
        self.class_names = names
        self.person_id, self.ball_id = resolve_class_ids(
            names, person_class_id, ball_class_id
        )
        if self.person_id is None:
            raise ValueError(
                f"Detector '{weights}' has no person/player class in {names}; "
                "pass person_class_id explicitly."
            )

    @property
    def name(self) -> str:
        return f"ultralytics:{self._weights}@{self._imgsz}"

    def predict(self, frame: np.ndarray) -> DetectionResult:
        results = self._model(
            frame, verbose=False, conf=self._conf, imgsz=self._imgsz
        )[0]
        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return DetectionResult.empty()

        cls_ids = boxes.cls.int().cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()

        person_mask = cls_ids == self.person_id
        person_xyxy = xyxy[person_mask]
        person_conf = conf[person_mask]

        ball_center: np.ndarray | None = None
        if self.ball_id is not None:
            ball_mask = cls_ids == self.ball_id
            if ball_mask.any():
                bb = xyxy[ball_mask]
                best = int(np.argmax(conf[ball_mask]))
                x1, y1, x2, y2 = bb[best]
                ball_center = np.array(
                    [(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=float
                )

        return DetectionResult(person_xyxy, person_conf, ball_center)


class DualDetector(BaseDetector):
    """
    Players from one backend, ball from another.

    WHY THIS EXISTS (measured, see Technical_Review_and_QA.md §13)
    -------------------------------------------------------------
    The two classes have opposite requirements and no single set of weights in
    this project satisfies both:

      * PLAYER is a large, canonical COCO object.  Stock COCO weights are
        domain-robust: `yolo11n.pt` returns 14-23 people/frame on ALL THREE
        source videos.  The 416-image Roboflow fine-tune, by contrast, learned
        the two courts it was annotated on and collapses on unseen footage -
        0.1 players/frame on "videoplayback (4)" (max class confidence 0.06 on
        a frame containing twelve clearly-visible players).  That single fact
        explains the all-TRANSITION demo renders: the tracker was never given
        anything to track.

      * BALL is a ~15 px, motion-blurred, non-canonical object.  Here the
        fine-tune wins decisively - 67% recall vs stock's 16% on the same
        video - because 233 annotated volleyballs beat COCO's "sports ball".

    Using each backend only where it measurably wins is therefore not a hack;
    it is the honest reading of the ablation.  Cost is one extra forward pass
    per frame, which can be halved with `ball_every_n > 1`.
    """

    def __init__(
        self,
        player_detector: BaseDetector,
        ball_detector: BaseDetector | None = None,
    ) -> None:
        self._players = player_detector
        self._ball = ball_detector
        self.class_names = getattr(player_detector, "class_names", {0: "person"})
        self.person_id = getattr(player_detector, "person_id", 0)
        # Advertise the ball capability of whichever backend actually supplies it.
        self.ball_id = (
            getattr(ball_detector, "ball_id", None) if ball_detector is not None
            else getattr(player_detector, "ball_id", None)
        )

    @property
    def name(self) -> str:
        if self._ball is None:
            return f"dual(players={self._players.name}, ball=none)"
        return f"dual(players={self._players.name}, ball={self._ball.name})"

    def predict(self, frame: np.ndarray) -> DetectionResult:
        det = self._players.predict(frame)
        if self._ball is None:
            return det
        ball = self._ball.predict(frame).ball_center
        # Keep the player backend's own ball only as a fallback: the dedicated
        # ball model is the authority whenever it fires.
        if ball is None:
            ball = det.ball_center
        return DetectionResult(
            person_xyxy=det.person_xyxy,
            person_conf=det.person_conf,
            ball_center=ball,
            keypoints=det.keypoints,
        )


def create_detector(
    weights: str,
    ball_weights: str | None = None,
    ball_conf_threshold: float | None = None,
    **kwargs,
) -> BaseDetector:
    """
    Factory.  Today every supported weights string routes to the ultralytics
    backend; a non-ultralytics backend (e.g. an ONNX or TensorRT engine) only
    needs a new BaseDetector subclass and one `elif` here – zero downstream
    changes (spec §5A).

    Passing `ball_weights` returns a DualDetector: `weights` supplies the
    players and `ball_weights` supplies the ball.  The ball backend runs at a
    lower confidence threshold by default, because a small fast object is
    recall-limited, not precision-limited - a spurious ball is repaired
    downstream by the trajectory gate, a missing one is not recoverable.
    """
    player_det = UltralyticsDetector(weights, **kwargs)
    if not ball_weights:
        return player_det

    ball_kwargs = dict(kwargs)
    ball_kwargs["conf_threshold"] = (
        ball_conf_threshold if ball_conf_threshold is not None
        else min(kwargs.get("conf_threshold", 0.35), 0.15)
    )
    # Class ids are resolved by name inside each backend; a ball-only model
    # legitimately has no person class, so tolerate that here.
    ball_kwargs.pop("person_class_id", None)
    try:
        ball_det: BaseDetector | None = UltralyticsDetector(
            ball_weights, **ball_kwargs
        )
        if getattr(ball_det, "ball_id", None) is None:
            print(
                f"[WARNING] Ball model '{ball_weights}' exposes no ball class "
                f"({getattr(ball_det, 'class_names', {})}); ignoring it.",
                file=sys.stderr,
            )
            ball_det = None
    except ValueError:
        # No person class in the ball model: acceptable, retry person-agnostic.
        ball_kwargs["person_class_id"] = -1
        ball_det = UltralyticsDetector(ball_weights, **ball_kwargs)

    return DualDetector(player_det, ball_det)


# ---------------------------------------------------------------------------
# 2. Temporal model interface
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    """Backend-agnostic sequence-classification output."""
    label: str                     # predicted class name
    numeric_idx: int               # user-facing numeric index (0 = Unclassified)
    confidence: float              # softmax prob of the argmax class
    entropy: float                 # Shannon entropy of the softmax (bits)
    entropy_norm: float            # entropy / log2(C)  in [0, 1]
    is_anomaly: bool               # confidence below the deviation threshold
    probs: dict[str, float] = field(default_factory=dict)


def shannon_entropy_bits(probs: np.ndarray) -> float:
    """H(p) = -sum p_i log2 p_i.  High H == the play deviates from every
    tactical template the model knows (spec §4C)."""
    p = np.asarray(probs, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


class BaseTemporalClassifier(ABC):
    """Abstract sequence classifier over (seq_len, feature_dim) inputs."""

    @abstractmethod
    def classify(self, seq: np.ndarray) -> ClassificationResult:
        """Classify one feature sequence."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Architecture identifier for logs."""


class TorchTemporalClassifier(BaseTemporalClassifier):
    """
    Loads a train_mamba.py checkpoint and serves it behind the abstract
    interface.  The concrete architecture is chosen from the checkpoint's own
    'arch' tag ('mamba' | 'transformer'), so a benchmark swap is a pure
    retraining exercise – this class, pipeline.py and infer_mamba.py are
    untouched (spec §5B).

    Safety checks
    -------------
    * feature-version guard: refuses to silently serve a checkpoint whose
      input semantics do not match the current feature contract.
    * anomaly flag: max softmax prob < `anomaly_threshold` marks the sequence
      "Anomaly / Tactical Deviation" for analytical review (spec §4C).
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: "str | object" = "cpu",
        anomaly_threshold: float = 0.5,
        strict_features: bool = False,
    ) -> None:
        import torch
        from mamba_model import LABEL_NAMES, create_temporal_model
        from label_clips import LABEL_TO_INDEX

        self._torch = torch
        self._label_names = LABEL_NAMES
        self._label_to_index = LABEL_TO_INDEX
        self._tau = float(anomaly_threshold)
        self._device = torch.device(device) if isinstance(device, str) else device

        ckpt = torch.load(checkpoint_path, map_location=self._device, weights_only=False)
        ckpt_version = ckpt.get("feature_version")
        if ckpt_version != FEATURE_VERSION:
            msg = (
                f"[FEATURE-VERSION MISMATCH] checkpoint='{ckpt_version}' "
                f"pipeline='{FEATURE_VERSION}'. The model was trained on a "
                "DIFFERENT input representation; its predictions are invalid. "
                "Retrain with train_mamba.py."
            )
            if strict_features:
                raise ValueError(msg)
            print(f"[WARNING] {msg}", file=sys.stderr)

        saved_args = ckpt.get("args", {})
        self._arch = saved_args.get("arch", "mamba")
        self._model = create_temporal_model(
            self._arch,
            input_dim=saved_args.get("input_dim", MODEL_INPUT_DIM),
            d_model=saved_args.get("d_model", 64),
            n_layers=saved_args.get("n_layers", 4),
            d_state=saved_args.get("d_state", 16),
            d_conv=saved_args.get("d_conv", 4),
            num_classes=len(ckpt.get("label_names", LABEL_NAMES)),
            dropout=0.0,
        )
        self._model.load_state_dict(ckpt["model_state"])
        self._model.to(self._device).eval()
        self._mean = ckpt["norm_mean"].to(self._device)
        self._std = ckpt["norm_std"].to(self._device)
        self._ckpt_labels = list(ckpt.get("label_names", LABEL_NAMES))

    @property
    def name(self) -> str:
        return f"torch:{self._arch}"

    @property
    def anomaly_threshold(self) -> float:
        return self._tau

    def classify(self, seq: np.ndarray) -> ClassificationResult:
        torch = self._torch
        t = torch.from_numpy(np.asarray(seq, dtype=np.float32)).to(self._device)
        t = (t - self._mean) / self._std
        with torch.no_grad():
            logits = self._model(t.unsqueeze(0))
            probs_t = torch.softmax(logits, dim=-1)[0]
        probs = probs_t.cpu().numpy()

        i = int(probs.argmax())
        label = self._ckpt_labels[i]
        conf = float(probs[i])
        h = shannon_entropy_bits(probs)
        h_norm = h / max(math.log2(len(probs)), 1e-9)

        return ClassificationResult(
            label=label,
            numeric_idx=self._label_to_index.get(label, 0),
            confidence=conf,
            entropy=round(h, 4),
            entropy_norm=round(h_norm, 4),
            is_anomaly=conf < self._tau,
            probs={n: round(float(p), 4) for n, p in zip(self._ckpt_labels, probs)},
        )


class SklearnTemporalClassifier(BaseTemporalClassifier):
    """
    Serves a `ml_classifier.py` .joblib behind the same interface as the Mamba
    checkpoint, so swapping the learned tactical model into the live pipeline is
    a filename change and nothing else.

    Both classifiers receive the identical (29, 18) `perm_invariant_v2` model
    sequence from `pipeline.py`; the temporal aggregation into a fixed-length
    vector happens here, using the SAME function the training run used
    (`ml_classifier.features_from_model_sequence`) and the feature mode recorded
    inside the .joblib. Neither the column order nor the aggregation can drift
    between fit and serve.

    A classical model has no softmax. `predict_proba` supplies calibrated-ish
    class probabilities for every estimator in the zoo (SVMs are constructed
    with `probability=True`), and entropy/anomaly are computed from those, so
    the HUD, the anomaly flag and the CSV columns behave identically whichever
    backend is loaded.
    """

    def __init__(self, model_path: str, anomaly_threshold: float = 0.5) -> None:
        try:
            import joblib
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "joblib is required to load a scikit-learn tactical model "
                "(pip install joblib)"
            ) from exc

        import ml_classifier

        bundle = joblib.load(model_path)
        self._pipeline = bundle["pipeline"]
        self._labels = list(bundle["class_names"])
        self._mode = bundle.get("feature_mode", "invariant")
        self._model_name = bundle.get("model_name", "sklearn")
        self._tau = float(anomaly_threshold)
        self._featurise = ml_classifier.features_from_model_sequence

        stored = bundle.get("feature_version")
        if stored and stored != FEATURE_VERSION:
            print(
                f"[WARNING] Tactical model '{model_path}' was fitted against "
                f"feature contract '{stored}' but this build serves "
                f"'{FEATURE_VERSION}'. Predictions are not trustworthy - "
                f"refit with fyp/ml_classifier.py.",
                file=sys.stderr,
            )

        from label_clips import LABEL_TO_INDEX
        self._label_to_index = LABEL_TO_INDEX

    @property
    def name(self) -> str:
        return f"sklearn:{self._model_name}({self._mode})"

    @property
    def anomaly_threshold(self) -> float:
        return self._tau

    def classify(self, seq: np.ndarray) -> ClassificationResult:
        x = self._featurise(np.asarray(seq, dtype=float), self._mode)
        probs = np.asarray(
            self._pipeline.predict_proba(x.reshape(1, -1))[0], dtype=float
        )

        i = int(probs.argmax())
        label = self._labels[i]
        conf = float(probs[i])
        h = shannon_entropy_bits(probs)
        h_norm = h / max(math.log2(len(probs)), 1e-9)

        return ClassificationResult(
            label=label,
            numeric_idx=self._label_to_index.get(label, 0),
            confidence=conf,
            entropy=round(h, 4),
            entropy_norm=round(h_norm, 4),
            is_anomaly=conf < self._tau,
            probs={n: round(float(p), 4) for n, p in zip(self._labels, probs)},
        )


def create_temporal_classifier(path: str, device="cpu",
                               anomaly_threshold: float = 0.5
                               ) -> BaseTemporalClassifier:
    """
    Pick the backend from the file extension: `.joblib`/`.pkl` -> scikit-learn,
    anything else -> the torch checkpoint loader. Lets `pipeline.py` accept
    either model with no flag and no branching at the call site.
    """
    if str(path).lower().endswith((".joblib", ".pkl")):
        return SklearnTemporalClassifier(path, anomaly_threshold=anomaly_threshold)
    return TorchTemporalClassifier(
        path, device=device, anomaly_threshold=anomaly_threshold
    )


__all__ = [
    "DetectionResult", "BaseDetector", "UltralyticsDetector", "DualDetector",
    "create_detector", "DEFAULT_IMGSZ",
    "ClassificationResult", "BaseTemporalClassifier", "TorchTemporalClassifier",
    "SklearnTemporalClassifier", "create_temporal_classifier",
    "shannon_entropy_bits",
]
