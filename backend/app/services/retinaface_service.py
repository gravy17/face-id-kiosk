"""
app/services/retinaface_service.py

Abstract interface + SCRFD (ONNX Runtime) implementation for face
detection.

Responsibilities:
  - Detect all faces in an image
  - Enforce exactly one face (reject 0 or 2+)
  - Return the detected face region, landmarks, and an aligned crop ready
    for the recognition model
  - Abstract the detector so a different backend can be swapped in

NOTE: module kept named `retinaface_service` (and DetectedFace / FaceDetector
/ get_face_detector kept as the public API) since the rest of the app
imports from this path — only the concrete implementation changed, from
the `retina-face` PyPI package to a local SCRFD ONNX model.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from fastapi import HTTPException, status

from app.core.config import Settings, get_settings
from app.core.logger import get_logger
from app.ml.face_align import norm_crop
from app.ml.onnx_session import load_session
from app.ml.scrfd import SCRFD, Detection

logger = get_logger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────

@dataclass
class DetectedFace:
    """Result of a successful single-face detection."""
    facial_area:  dict          # {"x": int, "y": int, "w": int, "h": int}
    landmarks:    dict          # {"left_eye": ..., "right_eye": ..., etc.}
    confidence:   float
    face_image:   np.ndarray    # 112x112 aligned BGR crop, ready for recognition


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


# ── SCRFD implementation ──────────────────────────────────────────────────

_LANDMARK_NAMES = ["left_eye", "right_eye", "nose", "mouth_left", "mouth_right"]


class SCRFDDetector(FaceDetector):

    def __init__(
        self,
        model_path,
        providers: list[str],
        input_size: int,
        conf_threshold: float,
        nms_threshold: float,
    ) -> None:
        session = load_session(model_path, providers)
        self._model = SCRFD(session, input_size=input_size)
        self._conf_threshold = conf_threshold
        self._nms_threshold = nms_threshold

    @property
    def detector_name(self) -> str:
        return "SCRFD"

    def detect_single_face(self, image: np.ndarray) -> DetectedFace:
        """
        Run SCRFD detection on the image.
        Raises HTTP 400 with a specific message for:
          - No face detected
          - Multiple faces detected
        """
        try:
            detections: list[Detection] = self._model.detect(
                image,
                conf_threshold=self._conf_threshold,
                nms_threshold=self._nms_threshold,
            )
        except Exception as exc:
            logger.error("SCRFD detection error", error=str(exc))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Face detection failed due to an internal error.",
            ) from exc

        if not detections:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No face detected in the image. "
                       "Please ensure your face is clearly visible and well-lit.",
            )

        if len(detections) > 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"{len(detections)} faces detected. "
                       "Only single-face images are accepted.",
            )

        det = detections[0]

        x1, y1, x2, y2 = det.bbox
        x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
        x2 = min(image.shape[1], int(round(x2)))
        y2 = min(image.shape[0], int(round(y2)))

        facial_area_dict = {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}
        landmarks_dict = {
            name: {"x": float(pt[0]), "y": float(pt[1])}
            for name, pt in zip(_LANDMARK_NAMES, det.landmarks)
        }

        # Align to a canonical 112x112 crop using the 5 keypoints — this is
        # what the ArcFace-family recognition model expects as input,
        # rather than a plain bbox crop+resize.
        aligned_face = norm_crop(image, det.landmarks, image_size=112)

        logger.debug(
            "Face detected",
            detector=self.detector_name,
            confidence=round(det.score, 4),
            area=facial_area_dict,
        )

        return DetectedFace(
            facial_area=facial_area_dict,
            landmarks=landmarks_dict,
            confidence=det.score,
            face_image=aligned_face,
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
        settings: Settings = get_settings()
        _detector_instance = SCRFDDetector(
            model_path=settings.detection_model_path(),
            providers=settings.ONNX_PROVIDERS,
            input_size=settings.DETECTION_INPUT_SIZE,
            conf_threshold=settings.DETECTION_CONF_THRESHOLD,
            nms_threshold=settings.DETECTION_NMS_THRESHOLD,
        )
        logger.info("FaceDetector initialised", detector=_detector_instance.detector_name)
    return _detector_instance
