"""
app/services/liveness_service.py

Abstract interface + MiniFASNetV2 (ONNX Runtime) implementation for
liveness / anti-spoofing.

MiniFASNet (Silent-Face-Anti-Spoofing architecture) classifies each face
crop into 3 classes; by the original project's convention, class index 1
is "real" and the rest ("spoof" / "unclear") are treated as not-live.

NOTE: module kept named `liveness_service` (and LivenessDetector /
get_liveness_detector kept as the public API) since the rest of the app
imports from this path — only the concrete implementation changed, from
the `Silent-Face-Anti-Spoofing` git package (which was never actually
installed, so this previously failed open on every check) to a local
MiniFASNetV2 ONNX model run directly through onnxruntime.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from fastapi import HTTPException, status

from app.core.config import Settings, get_settings
from app.core.logger import get_logger
from app.ml.liveness_crop import crop_for_liveness
from app.ml.onnx_session import load_session

logger = get_logger(__name__)

_REAL_CLASS_INDEX = 1  # matches the Silent-Face-Anti-Spoofing label convention


# ── Data class ────────────────────────────────────────────────────────────

@dataclass
class LivenessResult:
    passed:     bool
    score:      float   # 0.0–1.0, higher = more likely real
    label:      str     # "real" or "spoof"
    model_name: str


# ── Abstract interface ────────────────────────────────────────────────────

class LivenessDetector(ABC):
    """
    Abstract liveness / anti-spoofing detector.
    Implementations receive the full image (not just the crop) because
    MiniFASNet works best with context around the face.
    """

    @abstractmethod
    def check_liveness(
        self,
        image: np.ndarray,
        facial_area: dict,
    ) -> LivenessResult:
        """
        Run liveness check on image.
        facial_area: {"x": int, "y": int, "w": int, "h": int}
        Returns LivenessResult with passed=True only if the face is live.
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


# ── MiniFASNetV2 (ONNX) implementation ────────────────────────────────────

class MiniFASNetONNXDetector(LivenessDetector):
    """
    Wraps a single MiniFASNetV2 ONNX model. The reference implementation
    ensembles two model variants (a "2.7"-scale crop and a "4"/"1"-scale
    crop) and averages their scores; here we only have one exported model,
    so a single crop/score pass is used. See Settings.LIVENESS_CROP_SCALE
    if MiniFASNetV2.onnx was trained on a different crop convention.
    """

    def __init__(
        self,
        model_path,
        providers: list[str],
        input_size: int,
        crop_scale: float,
        threshold: float,
    ) -> None:
        self._session = load_session(model_path, providers)
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
        self._input_size = input_size
        self._crop_scale = crop_scale
        self._threshold = threshold
        self._model_name = "MiniFASNetV2"

    @property
    def model_name(self) -> str:
        return self._model_name

    def check_liveness(
        self,
        image: np.ndarray,
        facial_area: dict,
    ) -> LivenessResult:
        try:
            x, y, w, h = (
                facial_area["x"], facial_area["y"],
                facial_area["w"], facial_area["h"],
            )
            bbox = (x, y, x + w, y + h)

            crop = crop_for_liveness(
                image, bbox, scale=self._crop_scale, out_size=self._input_size,
            )

            # Silent-Face-Anti-Spoofing preprocessing: BGR, [0, 1] scale,
            # CHW, no channel swap and no per-channel normalisation.
            blob = crop.astype(np.float32) / 255.0
            blob = np.transpose(blob, (2, 0, 1))[None, ...]

            logits = self._session.run([self._output_name], {self._input_name: blob})[0]
            probs = _softmax(logits.reshape(-1))

            real_score = float(probs[_REAL_CLASS_INDEX])
            passed = real_score >= self._threshold

            logger.debug(
                "Liveness check complete",
                score=round(real_score, 4),
                threshold=self._threshold,
                passed=passed,
            )

            return LivenessResult(
                passed=passed,
                score=round(real_score, 4),
                label="real" if passed else "spoof",
                model_name=self._model_name,
            )

        except Exception as exc:
            logger.error("Liveness check error", error=str(exc))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Liveness check failed due to an internal error.",
            ) from exc


# ── Stub for development (no model files required) ────────────────────────

class StubLivenessDetector(LivenessDetector):
    """
    Development stub — always returns passed=True with a fixed score.
    Used automatically when the configured liveness model file is missing,
    so local dev doesn't require the ONNX weights to be present.
    """

    @property
    def model_name(self) -> str:
        return "StubLiveness"

    def check_liveness(self, image: np.ndarray, facial_area: dict) -> LivenessResult:
        logger.warning("StubLivenessDetector in use — not suitable for production")
        return LivenessResult(
            passed=True,
            score=0.99,
            label="real",
            model_name=self.model_name,
        )


# ── Factory ───────────────────────────────────────────────────────────────

_liveness_instance: LivenessDetector | None = None


def get_liveness_detector() -> LivenessDetector:
    """
    Returns the liveness detector instance.
    Uses MiniFASNetONNXDetector if the configured model file exists,
    StubLivenessDetector otherwise.
    """
    global _liveness_instance
    if _liveness_instance is None:
        settings: Settings = get_settings()
        model_path = settings.liveness_model_path()

        if model_path.exists():
            _liveness_instance = MiniFASNetONNXDetector(
                model_path=model_path,
                providers=settings.ONNX_PROVIDERS,
                input_size=settings.LIVENESS_INPUT_SIZE,
                crop_scale=settings.LIVENESS_CROP_SCALE,
                threshold=settings.LIVENESS_THRESHOLD,
            )
            logger.info("LivenessDetector initialised", model="MiniFASNetV2")
        else:
            logger.warning(
                "Liveness model not found — using stub liveness detector",
                model_path=str(model_path),
            )
            _liveness_instance = StubLivenessDetector()

    return _liveness_instance
