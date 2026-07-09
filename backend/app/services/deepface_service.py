"""
app/services/deepface_service.py

Abstract interface + DeepFace implementation for:
  - Generating face embeddings (registration)
  - Computing distance between embeddings (verification)

The interface decouples the rest of the app from DeepFace specifically —
swap the implementation without changing any calling code.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from deepface import DeepFace
from fastapi import HTTPException, status

from app.core.config import Settings, get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────

@dataclass
class EmbeddingResult:
    embedding:  list[float]
    model_name: str
    dimensions: int


@dataclass
class VerificationResult:
    distance:   float
    threshold:  float
    verified:   bool
    confidence: float   # 0–100, derived from distance vs threshold


# ── Abstract interface ────────────────────────────────────────────────────

class FaceEmbedder(ABC):
    """
    Abstract face embedding and verification service.
    Implementations must be stateless and thread-safe.
    """

    @abstractmethod
    def generate_embedding(self, face_image: np.ndarray) -> EmbeddingResult:
        """
        Generate a fixed-length embedding vector from a face image.
        face_image is a BGR numpy array of the cropped face region.
        """
        ...

    @abstractmethod
    def compute_distance(
        self,
        probe: list[float],
        reference: list[float],
        threshold: float,
    ) -> VerificationResult:
        """
        Compute distance between two embeddings and return a VerificationResult.
        Does NOT query the database — pure computation.
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """e.g. "Facenet512" """
        ...


# ── DeepFace implementation ───────────────────────────────────────────────

class DeepFaceEmbedder(FaceEmbedder):

    def __init__(self, settings: Settings) -> None:
        self._model       = settings.DEEPFACE_MODEL
        self._metric      = settings.DEEPFACE_DISTANCE_METRIC
        self._detector    = settings.RETINAFACE_DETECTOR

    @property
    def model_name(self) -> str:
        return self._model

    def generate_embedding(self, face_image: np.ndarray) -> EmbeddingResult:
        """
        Use DeepFace to extract an embedding from an already-cropped face image.
        detector_backend is set to "skip" since RetinaFace already ran detection —
        we don't want to run detection twice.
        """
        try:
            result = DeepFace.represent(
                img_path=face_image,
                model_name=self._model,
                detector_backend="skip",   # face already cropped by RetinaFace
                enforce_detection=False,
            )
        except Exception as exc:
            logger.error("DeepFace embedding error", error=str(exc))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Embedding generation failed.",
            ) from exc

        embedding = result[0]["embedding"]

        logger.debug(
            "Embedding generated",
            model=self._model,
            dimensions=len(embedding),
        )

        return EmbeddingResult(
            embedding=embedding,
            model_name=self._model,
            dimensions=len(embedding),
        )

    def compute_distance(
        self,
        probe: list[float],
        reference: list[float],
        threshold: float,
    ) -> VerificationResult:
        """
        Cosine distance between two normalised embedding vectors.
        Confidence is a linear score derived from the margin between
        the distance and the threshold.
        """
        probe_arr = np.array(probe,     dtype=np.float32)
        ref_arr   = np.array(reference, dtype=np.float32)

        # Normalise
        probe_arr /= np.linalg.norm(probe_arr) + 1e-10
        ref_arr   /= np.linalg.norm(ref_arr)   + 1e-10

        distance  = float(1.0 - np.dot(probe_arr, ref_arr))
        verified  = distance <= threshold

        # Confidence: 100 = perfect match (distance=0),
        #             0   = at threshold, negative beyond
        confidence = max(0.0, round((1.0 - distance / threshold) * 100, 2))

        logger.debug(
            "Distance computed",
            distance=round(distance, 4),
            threshold=threshold,
            verified=verified,
            confidence=confidence,
        )

        return VerificationResult(
            distance=round(distance, 4),
            threshold=threshold,
            verified=verified,
            confidence=confidence,
        )


# ── Factory ───────────────────────────────────────────────────────────────

_embedder_instance: FaceEmbedder | None = None


def get_face_embedder() -> FaceEmbedder:
    """
    Returns the singleton embedder instance.
    Change the returned type here to swap embedding implementations.
    """
    global _embedder_instance
    if _embedder_instance is None:
        settings          = get_settings()
        _embedder_instance = DeepFaceEmbedder(settings)
        logger.info("FaceEmbedder initialised", model=_embedder_instance.model_name)
    return _embedder_instance
