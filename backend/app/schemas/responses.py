"""
app/schemas/responses.py
Pydantic response models matching the agreed API response shapes exactly.
All responses include requestId, status, and timing.
"""
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ── Shared ────────────────────────────────────────────────────────────────

class BaseResponse(BaseModel):
    request_id:   str = Field(..., alias="requestId")
    status:       str

    model_config = {"populate_by_name": True}


# ── Challenge ─────────────────────────────────────────────────────────────

class ChallengeResponse(BaseModel):
    nonce:      str
    expires_at: int   # unix timestamp


# ── Registration ──────────────────────────────────────────────────────────

class RegistrationResponse(BaseResponse):
    user_id:          str   = Field(..., alias="userId")
    embedding_model:  str   = Field(..., alias="embeddingModel")
    detector:         str
    processing_time_ms: int = Field(..., alias="processingTimeMs")


# ── Verification ──────────────────────────────────────────────────────────

class MatchDetail(BaseModel):
    verified:   bool
    distance:   float | None
    threshold:  float
    confidence: float | None   # 0–100, derived from distance

class LivenessDetail(BaseModel):
    passed: bool
    score:  float | None

class ModelDetail(BaseModel):
    recognition: str
    detector:    str

class PerformanceDetail(BaseModel):
    execution_ms: int = Field(..., alias="executionMs")
    model_config  = {"populate_by_name": True}


class VerificationResponse(BaseModel):
    request_id:  str            = Field(..., alias="requestId")
    match:       MatchDetail
    liveness:    LivenessDetail
    model:       ModelDetail
    performance: PerformanceDetail
    token:       str | None     = None   # JWT, only present when verified=True

    model_config = {"populate_by_name": True}


# ── Fact Sheet ────────────────────────────────────────────────────────────

class FactSheetResponse(BaseModel):
    request_id:  str  = Field(..., alias="requestId")
    user_id:     str  = Field(..., alias="userId")
    first_name:  str  = Field(..., alias="firstName")
    last_name:   str  = Field(..., alias="lastName")
    nickname:    str | None
    favorite_color: str | None = Field(None, alias="favoriteColor")
    favorite_food:  str | None = Field(None, alias="favoriteFood")
    pet_name:    str | None    = Field(None, alias="petName")
    created_at:  datetime      = Field(..., alias="createdAt")
    verified:    bool          = True

    model_config = {"populate_by_name": True}


# ── Audit Logs ────────────────────────────────────────────────────────────

class AuditLogEntry(BaseModel):
    id:           str
    request_id:   str       = Field(..., alias="requestId")
    action:       str
    timestamp:    datetime
    status:       str
    client_ip:    str       = Field(..., alias="clientIp")
    execution_ms: int | None = Field(None, alias="executionMs")
    details:      Any | None = None

    model_config = {"populate_by_name": True}


class AuditLogResponse(BaseModel):
    request_id: str         = Field(..., alias="requestId")
    total:      int
    limit:      int
    offset:     int
    logs:       list[AuditLogEntry]

    model_config = {"populate_by_name": True}


# ── Health ────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:      str
    version:     str
    environment: str


# ── Error ─────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    request_id: str  = Field(..., alias="requestId")
    status:     str  = "error"
    detail:     str
    code:       str | None = None

    model_config = {"populate_by_name": True}
