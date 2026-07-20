"""
app/services/deepface_service.py

Abstract interface + ArcFace (ONNX Runtime, w600k_mbf) implementation for:
  - Generating face embeddings (registration)
  - Computing distance between embeddings (verification)

The interface decouples the rest of the app from the embedding backend —
swap the implementation without changing any calling code.

NOTE: module kept named `deepface_service` (and FaceEmbedder /
get_face_embedder kept as the public API) since the rest of the app
imports from this path — only the concrete implementation changed, from
the `deepface` PyPI package to a local ArcFace ONNX model.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import cv2
import numpy as np
from fastapi import HTTPException, status

from app.core.config import Settings, get_settings
from app.core.logger import get_logger
from app.ml.onnx_session import load_session

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
        face_image is a BGR numpy array — expected to already be the
        aligned crop produced by the face detector.
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
        """e.g. "w600k_mbf" """
        ...


# ── ArcFace (ONNX) implementation ─────────────────────────────────────────

class ArcFaceONNXEmbedder(FaceEmbedder):

    def __init__(self, model_path, providers: list[str], input_size: int) -> None:
        self._session = load_session(model_path, providers)
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
        self._input_size = input_size
        self._model_name = "w600k_mbf"

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate_embedding(self, face_image: np.ndarray) -> EmbeddingResult:
        """
        Run the ArcFace ONNX model on an already-aligned face crop.
        If the crop isn't exactly input_size x input_size (e.g. a caller
        passed a raw bbox crop instead of an aligned one), it's resized —
        alignment quality still matters for accuracy, but this keeps the
        method defensive rather than failing outright.
        """
        try:
            img = face_image
            if img.shape[0] != self._input_size or img.shape[1] != self._input_size:
                img = cv2.resize(img, (self._input_size, self._input_size))

            blob = cv2.dnn.blobFromImage(
                img,
                scalefactor=1.0 / 127.5,
                size=(self._input_size, self._input_size),
                mean=(127.5, 127.5, 127.5),
                swapRB=True,
            )

            output = self._session.run([self._output_name], {self._input_name: blob})[0]
            embedding = output.reshape(-1).astype(np.float32).tolist()

        except Exception as exc:
            logger.error("ArcFace embedding error", error=str(exc))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Embedding generation failed.",
            ) from exc

        logger.debug(
            "Embedding generated",
            model=self._model_name,
            dimensions=len(embedding),
        )

        return EmbeddingResult(
            embedding=embedding,
            model_name=self._model_name,
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
        settings: Settings = get_settings()
        _embedder_instance = ArcFaceONNXEmbedder(
            model_path=settings.recognition_model_path(),
            providers=settings.ONNX_PROVIDERS,
            input_size=settings.RECOGNITION_INPUT_SIZE,
        )
        logger.info("FaceEmbedder initialised", model=_embedder_instance.model_name)
    return _embedder_instance
