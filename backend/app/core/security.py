"""
app/core/security.py
Handles:
  - Short-lived JWT issuance and verification (fact sheet access)
  - Admin API key verification
  - Signed challenge nonce (webcam-only enforcement)
"""
import hashlib
import hmac
import time
import uuid
from typing import Any

from fastapi import Depends, Header, HTTPException, status
from jose import JWTError, jwt

from app.core.config import Settings, get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)


# ── JWT ────────────────────────────────────────────────────────────────────

def issue_fact_sheet_token(user_id: str, settings: Settings) -> str:
    """
    Issue a short-lived JWT granting access to a specific user's fact sheet.
    Expires after JWT_EXPIRY_MINUTES (default 15 min).
    """
    now = int(time.time())
    payload = {
        "sub":  user_id,
        "iat":  now,
        "exp":  now + (settings.JWT_EXPIRY_MINUTES * 60),
        "type": "factsheet",
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def verify_fact_sheet_token(token: str, settings: Settings) -> str:
    """
    Verify a fact sheet JWT. Returns the user_id (sub claim) on success.
    Raises HTTP 401 on any failure.
    """
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET,
            algorithms=[settings.JWT_ALGORITHM],
        )
        if payload.get("type") != "factsheet":
            raise JWTError("Wrong token type")
        user_id: str = payload["sub"]
        return user_id
    except JWTError as exc:
        logger.warning("JWT verification failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is invalid or has expired.",
        ) from exc


# ── Admin API Key ──────────────────────────────────────────────────────────

def verify_admin_key(
    authorization: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    """
    FastAPI dependency — validates the admin API key.
    Expects: Authorization: Bearer {ADMIN_API_KEY}
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin API key required.",
        )

    provided = authorization.removeprefix("Bearer ").strip()

    # Constant-time comparison to prevent timing attacks
    if not hmac.compare_digest(provided, settings.ADMIN_API_KEY):
        logger.warning("Invalid admin API key attempt")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid admin API key.",
        )


# ── Signed Challenge (webcam enforcement) ─────────────────────────────────

def issue_challenge(settings: Settings) -> dict[str, Any]:
    """
    Issues a one-time challenge nonce for the frontend to sign.
    The frontend captures a frame, appends the nonce + capture timestamp,
    and sends the HMAC-SHA256 signature alongside the image.

    Returns: { "nonce": str, "expires_at": int (unix timestamp) }
    """
    nonce      = uuid.uuid4().hex
    expires_at = int(time.time()) + settings.CHALLENGE_TTL_SECONDS
    return {"nonce": nonce, "expires_at": expires_at}


def verify_challenge_signature(
    nonce: str,
    capture_timestamp: int,
    signature: str,
    settings: Settings,
) -> None:
    """
    Verifies the HMAC-SHA256 signature the frontend sends with each image.

    The frontend computes:
        HMAC-SHA256(secret=CHALLENGE_SECRET, message=f"{nonce}:{capture_timestamp}")

    Raises HTTP 400 if the signature is invalid or the nonce has expired.
    """
    now = int(time.time())

    if now - capture_timestamp > settings.CHALLENGE_TTL_SECONDS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Challenge has expired. Request a fresh challenge and capture again.",
        )

    expected = hmac.new(
        key=settings.JWT_SECRET.encode(),
        msg=f"{nonce}:{capture_timestamp}".encode(),
        digestmod=hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(expected, signature):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid challenge signature. Image must be captured directly from webcam.",
        )
