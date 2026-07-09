"""
app/utils/image_validation.py
All pre-processing image checks before any ML model sees the image.
Each check raises a descriptive HTTPException on failure.

Checks (in order):
  1. MIME type (whitelist)
  2. File size
  3. Decode to numpy array
  4. Resolution bounds (min and max)
  5. Blur detection (Laplacian variance)

Face count checks (exactly one face) are done in retinaface_service.py
after detection, not here.
"""
import io
from dataclasses import dataclass

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile, status

from app.core.config import Settings, get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ValidatedImage:
    """Carries the decoded image array and its raw bytes after passing all checks."""
    array:      np.ndarray   # BGR, HxWxC
    raw_bytes:  bytes
    mime_type:  str
    width:      int
    height:     int
    sharpness:  float


async def validate_image(
    file: UploadFile,
    settings: Settings | None = None,
) -> ValidatedImage:
    """
    Run all image validation checks and return a ValidatedImage on success.
    Raises HTTP 400/413/415 with a descriptive message on any failure.
    """
    cfg = settings or get_settings()

    # ── 1. MIME type ──────────────────────────────────────────────────────
    content_type = file.content_type or ""
    if content_type not in cfg.ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported image type '{content_type}'. "
                f"Accepted: {', '.join(cfg.ALLOWED_MIME_TYPES)}"
            ),
        )

    # ── 2. File size ──────────────────────────────────────────────────────
    raw_bytes = await file.read()
    size      = len(raw_bytes)

    if size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image file is empty.",
        )

    if size > cfg.MAX_IMAGE_SIZE_BYTES:
        max_mb = cfg.MAX_IMAGE_SIZE_BYTES / (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image exceeds maximum size of {max_mb:.0f} MB.",
        )

    # ── 3. Decode ─────────────────────────────────────────────────────────
    arr = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image could not be decoded. Ensure it is a valid JPEG, PNG, or WebP.",
        )

    height, width = arr.shape[:2]

    # ── 4. Resolution bounds ──────────────────────────────────────────────
    if width < cfg.MIN_IMAGE_WIDTH or height < cfg.MIN_IMAGE_HEIGHT:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Image resolution {width}×{height} is below the minimum "
                f"{cfg.MIN_IMAGE_WIDTH}×{cfg.MIN_IMAGE_HEIGHT}."
            ),
        )

    if width > cfg.MAX_IMAGE_WIDTH or height > cfg.MAX_IMAGE_HEIGHT:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Image resolution {width}×{height} exceeds maximum "
                f"{cfg.MAX_IMAGE_WIDTH}×{cfg.MAX_IMAGE_HEIGHT}."
            ),
        )

    # ── 5. Blur detection (Laplacian variance) ────────────────────────────
    gray      = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if sharpness < cfg.MIN_SHARPNESS_SCORE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Image is too blurry (sharpness score {sharpness:.1f}, "
                f"minimum {cfg.MIN_SHARPNESS_SCORE}). "
                "Please capture a clearer image."
            ),
        )

    logger.debug(
        "Image validation passed",
        mime=content_type,
        size_bytes=size,
        resolution=f"{width}x{height}",
        sharpness=round(sharpness, 2),
    )

    return ValidatedImage(
        array=arr,
        raw_bytes=raw_bytes,
        mime_type=content_type,
        width=width,
        height=height,
        sharpness=sharpness,
    )
