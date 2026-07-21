"""
app/database/models.py
SQLAlchemy ORM models matching the agreed schema exactly.
All tables use UUIDs as primary keys (stored as strings in SQLite).
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id:             Mapped[str]      = mapped_column(String(36), primary_key=True, default=_uuid)
    first_name:     Mapped[str]      = mapped_column(String(100), nullable=False)
    last_name:      Mapped[str]      = mapped_column(String(100), nullable=False)
    nickname:       Mapped[str|None] = mapped_column(String(100), nullable=True)
    favorite_color: Mapped[str|None] = mapped_column(String(50),  nullable=True)
    favorite_food:  Mapped[str|None] = mapped_column(String(100), nullable=True)
    pet_name:       Mapped[str|None] = mapped_column(String(100), nullable=True)
    created_at:     Mapped[datetime] = mapped_column(default=_now)

    # One user → one FaceRecord (enforced at repository layer)
    face_record:       Mapped["FaceRecord|None"]    = relationship(
        "FaceRecord", back_populates="user", uselist=False, cascade="all, delete-orphan"
    )
    verification_logs: Mapped[list["VerificationLog"]] = relationship(
        "VerificationLog", back_populates="user", cascade="all, delete-orphan"
    )


class FaceRecord(Base):
    __tablename__ = "face_records"

    id:            Mapped[str]      = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id:       Mapped[str]      = mapped_column(ForeignKey("users.id"), nullable=False, unique=True)
    image_path:    Mapped[str]      = mapped_column(String(500), nullable=False)

    # JSON-serialised float list — interface abstracts this so a vector DB
    # can store it natively when swapped in
    embedding:     Mapped[str]      = mapped_column(Text, nullable=False)

    model_name:    Mapped[str]      = mapped_column(String(100), nullable=False)
    detector_name: Mapped[str]      = mapped_column(String(100), nullable=False)
    created_at:    Mapped[datetime] = mapped_column(default=_now)

    user: Mapped["User"] = relationship("User", back_populates="face_record")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id:           Mapped[str]      = mapped_column(String(36), primary_key=True, default=_uuid)
    request_id:   Mapped[str]      = mapped_column(String(50),  nullable=False, index=True)
    action:       Mapped[str]      = mapped_column(String(50),  nullable=False)
    timestamp:    Mapped[datetime] = mapped_column(default=_now, index=True)
    status:       Mapped[str]      = mapped_column(String(20),  nullable=False)
    client_ip:    Mapped[str]      = mapped_column(String(50),  nullable=False)
    execution_ms: Mapped[int|None] = mapped_column(Integer,     nullable=True)

    # JSON blob for endpoint-specific extras (model used, scores, errors, etc.)
    details:      Mapped[str|None] = mapped_column(Text, nullable=True)


class VerificationLog(Base):
    __tablename__ = "verification_logs"

    id:             Mapped[str]        = mapped_column(String(36), primary_key=True, default=_uuid)
    request_id:     Mapped[str]        = mapped_column(String(50),  nullable=False, index=True)
    user_id:        Mapped[str|None]   = mapped_column(ForeignKey("users.id"), nullable=True)
    distance:       Mapped[float|None] = mapped_column(Float,   nullable=True)
    threshold:      Mapped[float|None] = mapped_column(Float,   nullable=True)
    verified:       Mapped[bool]       = mapped_column(Boolean, nullable=False)
    liveness_score: Mapped[float|None] = mapped_column(Float,   nullable=True)
    created_at:     Mapped[datetime]   = mapped_column(default=_now)

    user: Mapped["User|None"] = relationship("User", back_populates="verification_logs")


class Challenge(Base):
    """
    Server-issued, single-use nonce backing the webcam-capture 'freshness'
    check on /register and /verify. Replaces the old client-side HMAC
    signature scheme (which required shipping a secret to the browser —
    see app/core/security.py for why that was unsound).
    """
    __tablename__ = "challenges"

    nonce:      Mapped[str]      = mapped_column(String(64), primary_key=True)
    expires_at: Mapped[datetime] = mapped_column(nullable=False, index=True)
    used:       Mapped[bool]     = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=_now)
