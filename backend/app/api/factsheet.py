"""
app/api/factsheet.py
GET /factsheet/{userId}  — returns user fact sheet (JWT protected)
GET /logs                — returns audit logs (admin key protected)
GET /health              — public health check
POST /challenge          — issues a signed challenge nonce for webcam enforcement
DELETE /admin/cleanup    — manual cleanup trigger (admin key protected)
"""
import json
import uuid

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.logger import get_logger, request_id_var
from app.core.security import verify_admin_key, verify_fact_sheet_token, issue_challenge
from app.database.db import get_db
from app.database.repository import FacialKioskRepository, get_repository
from app.schemas.responses import (
    AuditLogEntry,
    AuditLogResponse,
    ChallengeResponse,
    FactSheetResponse,
    HealthResponse,
)

logger = get_logger(__name__)
router = APIRouter(tags=["Fact Sheet & Admin"])


# ── Challenge ─────────────────────────────────────────────────────────────

@router.post(
    "/challenge",
    response_model=ChallengeResponse,
    summary="Request a signed challenge nonce for webcam enforcement",
)
async def get_challenge(settings: Settings = Depends(get_settings)) -> ChallengeResponse:
    challenge = issue_challenge(settings)
    return ChallengeResponse(**challenge)


# ── Fact Sheet ────────────────────────────────────────────────────────────

@router.get(
    "/factsheet/{user_id}",
    response_model=FactSheetResponse,
    summary="Retrieve a user's fact sheet (requires valid verification token)",
)
async def get_fact_sheet(
    user_id:       str,
    authorization: str | None = Header(default=None),
    db:            AsyncSession = Depends(get_db),
    settings:      Settings     = Depends(get_settings),
) -> FactSheetResponse:

    request_id = request_id_var.get(f"REQ-{uuid.uuid4().hex[:8].upper()}")

    # ── Extract and verify JWT ────────────────────────────────────────────
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Verification token required. Complete facial verification first.",
        )

    token          = authorization.removeprefix("Bearer ").strip()
    token_user_id  = verify_fact_sheet_token(token, settings)

    # ── Token must be for this specific user ──────────────────────────────
    if token_user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Token is not valid for this user.",
        )

    # ── Fetch user ────────────────────────────────────────────────────────
    repo: FacialKioskRepository = get_repository(db)
    user = await repo.get_user_by_id(user_id)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    logger.info("Fact sheet accessed", user_id=user_id)

    return FactSheetResponse(
        requestId=request_id,
        userId=user.id,
        firstName=user.first_name,
        lastName=user.last_name,
        nickname=user.nickname,
        favoriteColor=user.favorite_color,
        favoriteFood=user.favorite_food,
        petName=user.pet_name,
        createdAt=user.created_at,
        verified=True,
    )


# ── Audit Logs ────────────────────────────────────────────────────────────

@router.get(
    "/logs",
    response_model=AuditLogResponse,
    summary="Retrieve audit logs (admin only)",
    dependencies=[Depends(verify_admin_key)],
)
async def get_logs(
    limit:  int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0,   ge=0),
    db:     AsyncSession = Depends(get_db),
) -> AuditLogResponse:

    request_id = request_id_var.get(f"REQ-{uuid.uuid4().hex[:8].upper()}")
    repo: FacialKioskRepository = get_repository(db)

    logs = await repo.get_audit_logs(limit=limit, offset=offset)

    entries = [
        AuditLogEntry(
            id=log.id,
            requestId=log.request_id,
            action=log.action,
            timestamp=log.timestamp,
            status=log.status,
            clientIp=log.client_ip,
            executionMs=log.execution_ms,
            details=json.loads(log.details) if log.details else None,
        )
        for log in logs
    ]

    return AuditLogResponse(
        requestId=request_id,
        total=len(entries),
        limit=limit,
        offset=offset,
        logs=entries,
    )


# ── Admin Cleanup ─────────────────────────────────────────────────────────

@router.delete(
    "/admin/cleanup",
    summary="Manually trigger deletion of records older than N days (admin only)",
    dependencies=[Depends(verify_admin_key)],
)
async def manual_cleanup(
    days:     int         = Query(default=None, description="Override CLEANUP_AGE_DAYS from config"),
    db:       AsyncSession = Depends(get_db),
    settings: Settings     = Depends(get_settings),
) -> dict:

    age = days if days is not None else settings.CLEANUP_AGE_DAYS
    repo: FacialKioskRepository = get_repository(db)
    deleted = await repo.delete_records_older_than(age)

    logger.info("Manual cleanup completed", deleted=deleted, age_days=age)
    return {"status": "success", "deleted": deleted, "age_days": age}


# ── Health ────────────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
)
async def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
    )
