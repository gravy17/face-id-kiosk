"""
app/api/register.py
POST /register — full registration pipeline:
  1. Validate challenge signature (webcam enforcement)
  2. Validate image (MIME, size, resolution, blur)
  3. Detect face with RetinaFace (exactly one face required)
  4. Check liveness with MiniFASNet
  5. Check for duplicate face against existing embeddings
  6. Generate embedding with DeepFace
  7. Save image to disk
  8. Persist user + face record
  9. Write audit log
  10. Return RegistrationResponse
"""
import uuid

from fastapi import APIRouter, Request, Depends, Form, HTTPException, UploadFile, status
from fastapi import File as FastAPIFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.logger import get_logger, request_id_var
from app.core.security import verify_challenge_signature
from app.database.db import get_db
from app.database.repository import FacialKioskRepository, get_repository
from app.schemas.responses import RegistrationResponse
from app.services.deepface_service import get_face_embedder
from app.services.image_service import get_image_service
from app.services.liveness_service import get_liveness_detector
from app.services.report_service import registration_details, rejection_details
from app.services.retinaface_service import get_face_detector
from app.utils.image_validation import validate_image
from app.utils.timing import timed
from slowapi import Limiter
from slowapi.util import get_remote_address

settings = get_settings()
logger = get_logger(__name__)
router = APIRouter(prefix="/register", tags=["Registration"])
limiter = Limiter(key_func=get_remote_address)

@router.post(
    "",
    response_model=RegistrationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user with facial capture",
)
@limiter.limit(settings.RATE_LIMIT_REGISTER)
async def register(
    request:           Request,
    # ── Image ──────────────────────────────────────────────────────────────
    image:             UploadFile = FastAPIFile(..., description="Webcam capture (JPEG/PNG/WebP)"),

    # ── Webcam challenge fields ────────────────────────────────────────────
    nonce:             str = Form(...),
    capture_timestamp: int = Form(...),
    signature:         str = Form(...),

    # ── User details ──────────────────────────────────────────────────────
    first_name:     str        = Form(...),
    last_name:      str        = Form(...),
    nickname:       str | None = Form(None),
    favorite_color: str | None = Form(None),
    favorite_food:  str | None = Form(None),
    pet_name:       str | None = Form(None),

    # ── Dependencies ──────────────────────────────────────────────────────
    db:       AsyncSession         = Depends(get_db),
    settings: Settings             = Depends(get_settings),
) -> RegistrationResponse:

    request_id = request_id_var.get(f"REQ-{uuid.uuid4().hex[:8].upper()}")
    repo: FacialKioskRepository = get_repository(db)

    with timed() as t:

        # ── 1. Verify webcam challenge ─────────────────────────────────────
        verify_challenge_signature(nonce, capture_timestamp, signature, settings)

        # ── 2. Validate image ─────────────────────────────────────────────
        validated = await validate_image(image, settings)

        # ── 3. Detect face ────────────────────────────────────────────────
        detector     = get_face_detector()
        detected     = detector.detect_single_face(validated.array)

        # ── 4. Liveness check ─────────────────────────────────────────────
        liveness_svc = get_liveness_detector()
        liveness     = liveness_svc.check_liveness(validated.array, detected.facial_area)

        if not liveness.passed:
            await repo.create_audit_log(
                request_id=request_id,
                action="REGISTER",
                status="REJECTED_LIVENESS",
                client_ip="-",
                execution_ms=t.elapsed_ms(),
                details=rejection_details(
                    "Liveness check failed",
                    liveness_score=liveness.score,
                    liveness_model=liveness.model_name,
                ),
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "message": "Liveness check failed. Please use a live camera — "
                               "photos and screen replays are not accepted.",
                    "liveness_score": liveness.score,
                },
            )

        # ── 5. Duplicate face check ────────────────────────────────────────
        embedder        = get_face_embedder()
        probe_embedding = embedder.generate_embedding(detected.face_image)

        existing = await repo.find_closest_match(
            probe_embedding.embedding,
            settings.VERIFICATION_THRESHOLD,
        )

        if existing:
            existing_user, distance = existing
            await repo.create_audit_log(
                request_id=request_id,
                action="REGISTER",
                status="REJECTED_DUPLICATE",
                client_ip="-",
                execution_ms=t.elapsed_ms(),
                details=rejection_details(
                    "Duplicate face detected",
                    matched_user_id=existing_user.id,
                    distance=distance,
                ),
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="A user with this face is already registered.",
            )

        # ── 6. Save image ─────────────────────────────────────────────────
        image_svc   = get_image_service()
        tmp_user_id = str(uuid.uuid4())   # temp ID for path before user creation
        image_path  = image_svc.save_face(tmp_user_id, detected.face_image)

        # ── 7. Persist user and face record ───────────────────────────────
        user = await repo.create_user(
            first_name=first_name,
            last_name=last_name,
            nickname=nickname,
            favorite_color=favorite_color,
            favorite_food=favorite_food,
            pet_name=pet_name,
        )

        await repo.upsert_face_record(
            user_id=user.id,
            image_path=image_path,
            embedding=probe_embedding.embedding,
            model_name=embedder.model_name,
            detector_name=detector.detector_name,
        )

        # ── 8. Audit log ──────────────────────────────────────────────────
        await repo.create_audit_log(
            request_id=request_id,
            action="REGISTER",
            status="SUCCESS",
            client_ip="-",
            execution_ms=t.elapsed_ms(),
            details=registration_details(
                user_id=user.id,
                detected_face=detected,
                embedding_result=probe_embedding,
                liveness_result=liveness,
                image_path=image_path,
            ),
        )

    logger.info(
        "Registration successful",
        user_id=user.id,
        model=embedder.model_name,
        elapsed_ms=t.elapsed_ms(),
    )

    return RegistrationResponse(
        requestId=request_id,
        status="success",
        userId=user.id,
        embeddingModel=embedder.model_name,
        detector=detector.detector_name,
        processingTimeMs=t.elapsed_ms(),
    )
