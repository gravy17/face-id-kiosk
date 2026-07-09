"""
app/services/retinaface_service.py

Abstract interface + RetinaFace implementation for face detection.

Responsibilities:
  - Detect all faces in an image
  - Enforce exactly one face (reject 0 or 2+)
  - Return the detected face region and landmarks
  - Abstract the detector so a different backend can be swapped in
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from fastapi import HTTPException, status
from retinaface import RetinaFace

from app.core.logger import get_logger

logger = get_logger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────

@dataclass
class DetectedFace:
    """Result of a successful single-face detection."""
    facial_area:  dict          # {"x": int, "y": int, "w": int, "h": int}
    landmarks:    dict          # {"right_eye": ..., "left_eye": ..., etc.}
    confidence:   float
    face_image:   np.ndarray    # cropped BGR face region


# ── Abstract interface ────────────────────────────────────────────────────

class FaceDetector(ABC):
    """
    Abstract face detector.
    Swap implementations by returning a different subclass
    from get_face_detector().
    """

    @abstractmethod
    def detect_single_face(self, image: np.ndarray) -> DetectedFace:
        """
        Detect faces in image. Must return exactly one DetectedFace.
        Raises HTTP 400 if zero or multiple faces are found.
        """
        ...

    @property
    @abstractmethod
    def detector_name(self) -> str:
        """Human-readable name for logging and response metadata."""
        ...


# ── RetinaFace implementation ─────────────────────────────────────────────

class RetinafaceDetector(FaceDetector):

    @property
    def detector_name(self) -> str:
        return "RetinaFace"

    def detect_single_face(self, image: np.ndarray) -> DetectedFace:
        """
        Run RetinaFace detection on the image.
        Raises HTTP 400 with a specific message for:
          - No face detected
          - Multiple faces detected
          - Low confidence detection
        """
        try:
            faces = RetinaFace.detect_faces(image)
        except Exception as exc:
            logger.error("RetinaFace detection error", error=str(exc))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Face detection failed due to an internal error.",
            ) from exc

        # RetinaFace returns a dict of faces or 0 (no faces)
        if not faces or not isinstance(faces, dict):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No face detected in the image. "
                       "Please ensure your face is clearly visible and well-lit.",
            )

        if len(faces) > 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"{len(faces)} faces detected. "
                       "Only single-face images are accepted.",
            )

        face_key  = next(iter(faces))
        face_data = faces[face_key]

        confidence = float(face_data.get("score", 0.0))
        facial_area = face_data.get("facial_area", {})  # [x1, y1, x2, y2]
        landmarks   = face_data.get("landmarks", {})

        # Crop face region from image
        x1, y1, x2, y2 = facial_area
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
        face_crop = image[y1:y2, x1:x2]

        facial_area_dict = {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}

        logger.debug(
            "Face detected",
            detector=self.detector_name,
            confidence=round(confidence, 4),
            area=facial_area_dict,
        )

        return DetectedFace(
            facial_area=facial_area_dict,
            landmarks=landmarks,
            confidence=confidence,
            face_image=face_crop,
        )


# ── Factory ───────────────────────────────────────────────────────────────

_detector_instance: FaceDetector | None = None


def get_face_detector() -> FaceDetector:
    """
    Returns the singleton detector instance.
    Change the returned type here to swap detector implementations.
    """
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = RetinafaceDetector()
        logger.info("FaceDetector initialised", detector=_detector_instance.detector_name)
    return _detector_instance
