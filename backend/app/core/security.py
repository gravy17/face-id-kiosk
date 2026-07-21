"""
app/core/security.py
Handles:
  - Short-lived JWT issuance and verification (fact sheet access)
  - Admin API key verification
  - Server-tracked challenge nonce (webcam-capture freshness / replay prevention)
"""
import hmac
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from fastapi import Depends, Header, HTTPException, status
from jose import JWTError, jwt

from app.core.config import Settings, get_settings
from app.core.logger import get_logger

if TYPE_CHECKING:
    from app.database.repository import FacialKioskRepository

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


# ── Signed Challenge → Server-Tracked Challenge (webcam enforcement) ───────
#
# Previously this issued a nonce and expected the frontend to return an
# HMAC-SHA256 signature over it, computed with a "shared secret" baked
# into the frontend build. That secret is not actually secret — any
# VITE_/REACT_APP_-style env var ends up inlined in the shipped JS bundle,
# readable by anyone via devtools. So the signature never proved anything
# beyond "this request was built by code that can read our own bundle" —
# which is not a meaningful barrier.
#
# This now tracks challenges server-side instead: /challenge issues a
# nonce and persists it with an expiry; /register and /verify redeem it
# via a single atomic "consume" operation (see
# SQLiteRepository.consume_challenge) that fails if the nonce is missing,
# expired, or already used. No secret ever reaches the browser, and reuse
# (replay) of a captured request is blocked because a nonce can only be
# consumed once.

async def issue_challenge(
    settings: Settings,
    repo: "FacialKioskRepository",
) -> dict[str, Any]:
    """
    Issues a one-time challenge nonce and persists it server-side.
    Returns: { "nonce": str, "expires_at": int (unix timestamp) }
    """
    nonce      = uuid.uuid4().hex
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=settings.CHALLENGE_TTL_SECONDS)
    await repo.create_challenge(nonce, expires_at)
    return {"nonce": nonce, "expires_at": int(expires_at.timestamp())}


async def consume_challenge(
    nonce: str,
    repo: "FacialKioskRepository",
) -> None:
    """
    Validates and consumes a challenge nonce. Raises HTTP 400 if the nonce
    is unknown, expired, or has already been used (replay).
    """
    ok = await repo.consume_challenge(nonce)
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid, expired, or already-used challenge. "
                   "Request a fresh challenge and capture again.",
        )
