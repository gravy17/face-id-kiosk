"""
app/api/verify.py
POST /verify — full verification pipeline:
  1. Validate challenge signature
  2. Validate image
  3. Detect face (RetinaFace)
  4. Liveness check (MiniFASNet)
  5. Generate probe embedding (DeepFace)
  6. Search all stored embeddings for closest match
  7. Issue JWT if verified
  8. Write audit and verification logs
  9. Return detailed VerificationResponse
"""
import uuid

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile, status
from fastapi import File as FastAPIFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.logger import get_logger, request_id_var
from app.core.security import issue_fact_sheet_token, verify_challenge_signature
from app.database.db import get_db
from app.database.repository import FacialKioskRepository, get_repository
from app.schemas.responses import (
    LivenessDetail,
    MatchDetail,
    ModelDetail,
    PerformanceDetail,
    VerificationResponse,
)
from app.services.deepface_service import get_face_embedder
from app.services.liveness_service import get_liveness_detector
from app.services.report_service import verification_details
from app.services.retinaface_service import get_face_detector
from app.utils.image_validation import validate_image
from app.utils.timing import timed

logger = get_logger(__name__)
router = APIRouter(prefix="/verify", tags=["Verification"])


@router.post(
    "",
    response_model=VerificationResponse,
    summary="Verify identity and receive a short-lived fact sheet token",
)
async def verify(
    image:             UploadFile = FastAPIFile(...),
    nonce:             str = Form(...),
    capture_timestamp: int = Form(...),
    signature:         str = Form(...),

    db:       AsyncSession = Depends(get_db),
    settings: Settings     = Depends(get_settings),
) -> VerificationResponse:

    request_id = request_id_var.get(f"REQ-{uuid.uuid4().hex[:8].upper()}")
    repo: FacialKioskRepository = get_repository(db)

    with timed() as t:

        # ── 1. Challenge verification ──────────────────────────────────────
        verify_challenge_signature(nonce, capture_timestamp, signature, settings)

        # ── 2. Image validation ────────────────────────────────────────────
        validated = await validate_image(image, settings)

        # ── 3. Face detection ──────────────────────────────────────────────
        detector = get_face_detector()
        detected = detector.detect_single_face(validated.array)

        # ── 4. Liveness check ──────────────────────────────────────────────
        liveness_svc = get_liveness_detector()
        liveness     = liveness_svc.check_liveness(validated.array, detected.facial_area)

        embedder    = get_face_embedder()
        token:      str | None = None
        match_user_id: str | None = None

        if not liveness.passed:
            # Log and return — no embedding search needed
            await _write_logs(
                repo, request_id, liveness, None, None, False,
                settings, t.elapsed_ms(),
            )
            return VerificationResponse(
                requestId=request_id,
                match=MatchDetail(
                    verified=False,
                    distance=None,
                    threshold=settings.VERIFICATION_THRESHOLD,
                    confidence=None,
                ),
                liveness=LivenessDetail(passed=False, score=liveness.score),
                model=ModelDetail(
                    recognition=embedder.model_name,
                    detector=detector.detector_name,
                ),
                performance=PerformanceDetail(executionMs=t.elapsed_ms()),
                token=None,
            )

        # ── 5. Generate probe embedding ────────────────────────────────────
        probe = embedder.generate_embedding(detected.face_image)

        # ── 6. Search for closest match ────────────────────────────────────
        match_result = await repo.find_closest_match(
            probe.embedding,
            settings.VERIFICATION_THRESHOLD,
        )

        if match_result:
            matched_user, distance = match_result
            match_user_id = matched_user.id

            ver_result = embedder.compute_distance(
                probe.embedding,
                [],   # distance already computed in find_closest_match
                settings.VERIFICATION_THRESHOLD,
            )
            # Override with the actual stored distance
            from dataclasses import replace
            ver_result = replace(ver_result, distance=round(distance, 4))

            # ── 7. Issue JWT ───────────────────────────────────────────────
            token = issue_fact_sheet_token(matched_user.id, settings)

            verified    = True
            confidence  = max(0.0, round((1.0 - distance / settings.VERIFICATION_THRESHOLD) * 100, 2))
        else:
            verified   = False
            distance   = None
            confidence = None

        await _write_logs(
            repo, request_id, liveness,
            distance, match_user_id, verified,
            settings, t.elapsed_ms(),
        )

    logger.info(
        "Verification complete",
        verified=verified,
        matched_user=match_user_id,
        elapsed_ms=t.elapsed_ms(),
    )

    return VerificationResponse(
        requestId=request_id,
        match=MatchDetail(
            verified=verified,
            distance=distance,
            threshold=settings.VERIFICATION_THRESHOLD,
            confidence=confidence,
        ),
        liveness=LivenessDetail(passed=liveness.passed, score=liveness.score),
        model=ModelDetail(
            recognition=embedder.model_name,
            detector=detector.detector_name,
        ),
        performance=PerformanceDetail(executionMs=t.elapsed_ms()),
        token=token,
    )


async def _write_logs(
    repo: FacialKioskRepository,
    request_id: str,
    liveness,
    distance: float | None,
    user_id: str | None,
    verified: bool,
    settings: Settings,
    elapsed_ms: int,
) -> None:
    """Write both audit log and verification log atomically."""
    ver_result_for_report = None
    if distance is not None:
        from app.services.deepface_service import VerificationResult
        ver_result_for_report = VerificationResult(
            distance=distance,
            threshold=settings.VERIFICATION_THRESHOLD,
            verified=verified,
            confidence=max(0.0, (1.0 - distance / settings.VERIFICATION_THRESHOLD) * 100),
        )

    await repo.create_audit_log(
        request_id=request_id,
        action="VERIFY",
        status="VERIFIED" if verified else ("LIVENESS_FAIL" if not liveness.passed else "NO_MATCH"),
        client_ip="-",
        execution_ms=elapsed_ms,
        details=verification_details(liveness, ver_result_for_report, user_id),
    )

    await repo.create_verification_log(
        request_id=request_id,
        verified=verified,
        user_id=user_id,
        distance=distance,
        threshold=settings.VERIFICATION_THRESHOLD,
        liveness_score=liveness.score,
    )
