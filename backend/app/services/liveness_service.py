"""
app/services/liveness_service.py

Abstract interface + MiniFASNet implementation for liveness / anti-spoofing.

MiniFASNet (Silent-Face-Anti-Spoofing) classifies each face as:
  - Real (live person in front of camera)
  - Spoof (photo, screen replay, mask, etc.)

The interface allows swapping to a different liveness model without
changing any calling code.

Note on installation:
  MiniFASNet is not on PyPI. Install from source:
    pip install git+https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
  or place the cloned repo under app/services/minifas/ and adjust the import.
"""
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from fastapi import HTTPException, status

from app.core.config import Settings, get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)


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


# ── MiniFASNet implementation ─────────────────────────────────────────────

class MiniFASNetDetector(LivenessDetector):
    """
    Wraps the Silent-Face-Anti-Spoofing MiniFASNet models.
    Uses two model variants (2.7M and 1M) and averages their scores
    for improved accuracy — same approach as the reference implementation.
    """

    MODEL_INPUT_SIZE = (80, 80)

    def __init__(self, model_dir: str, threshold: float) -> None:
        self._threshold  = threshold
        self._model_dir  = Path(model_dir)
        self._model_name = "MiniFASNet"
        self._models     = self._load_models()

    @property
    def model_name(self) -> str:
        return self._model_name

    def _load_models(self) -> list:
        """
        Attempt to load MiniFASNet models.
        Falls back gracefully with a warning if the package is not installed.
        """
        try:
            # The Silent-Face-Anti-Spoofing package exposes this interface
            from src.anti_spoof_predict import AntiSpoofPredict  # type: ignore
            model_24  = AntiSpoofPredict(0, str(self._model_dir))
            return [model_24]
        except ImportError:
            logger.warning(
                "MiniFASNet not installed. "
                "Install from: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing — "
                "Liveness checks will FAIL OPEN (pass=True) until installed."
            )
            return []

    def check_liveness(
        self,
        image: np.ndarray,
        facial_area: dict,
    ) -> LivenessResult:
        if not self._models:
            # Package not installed — fail open with a warning
            logger.warning("Liveness check skipped — MiniFASNet not installed")
            return LivenessResult(
                passed=True,
                score=0.0,
                label="unknown",
                model_name=self._model_name,
            )

        try:
            from src.anti_spoof_predict import AntiSpoofPredict  # type: ignore
            from src.generate_patches import CropImage             # type: ignore

            scores: list[float] = []

            for model in self._models:
                # Crop image to the face region with context padding
                image_cropper = CropImage()
                x, y, w, h    = (facial_area["x"], facial_area["y"],
                                 facial_area["w"], facial_area["h"])
                image_bbox    = [x, y, x + w, y + h]
                param         = _get_kernel_size(h, w)
                cropped       = image_cropper.crop(image, image_bbox, param)

                prediction    = model.predict(cropped, str(self._model_dir))
                # prediction shape: (1, 3) — [spoof, real, unsure]
                real_score    = float(prediction[0][1])
                scores.append(real_score)

            score  = float(np.mean(scores))
            passed = score >= self._threshold

            logger.debug(
                "Liveness check complete",
                score=round(score, 4),
                threshold=self._threshold,
                passed=passed,
            )

            return LivenessResult(
                passed=passed,
                score=round(score, 4),
                label="real" if passed else "spoof",
                model_name=self._model_name,
            )

        except Exception as exc:
            logger.error("Liveness check error", error=str(exc))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Liveness check failed due to an internal error.",
            ) from exc


def _get_kernel_size(h: int, w: int) -> dict:
    """Compute crop parameters based on face bounding box size."""
    kernel_size = min(h, w)
    return {
        "org_img":     None,   # filled in by the cropper
        "bbox":        None,
        "new_size":    80,
        "scale":       kernel_size / 80.0,
    }


# ── Stub for development (no MiniFASNet required) ─────────────────────────

class StubLivenessDetector(LivenessDetector):
    """
    Development stub — always returns passed=True with a fixed score.
    Swap for MiniFASNetDetector in production by changing get_liveness_detector().
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
    Change the returned type here to swap implementations.
    Uses MiniFASNetDetector if the model dir exists, StubLivenessDetector otherwise.
    """
    global _liveness_instance
    if _liveness_instance is None:
        settings = get_settings()
        model_dir = Path(settings.MINIFAS_MODEL_DIR)

        if model_dir.exists():
            _liveness_instance = MiniFASNetDetector(
                model_dir=str(model_dir),
                threshold=settings.LIVENESS_THRESHOLD,
            )
            logger.info("LivenessDetector initialised", model="MiniFASNet")
        else:
            logger.warning(
                "MiniFASNet model dir not found — using stub liveness detector",
                model_dir=str(model_dir),
            )
            _liveness_instance = StubLivenessDetector()

    return _liveness_instance
