"""
app/database/repository.py

Abstract repository interface + SQLite implementation.

The interface (FacialKioskRepository) is the seam for swapping in a
vector database later. The SQLite implementation stores embeddings as
JSON-serialised float lists. A vector DB implementation would store
them natively and override find_closest_match with an ANN query.
"""
import json
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logger import get_logger
from app.database.models import AuditLog, FaceRecord, User, VerificationLog

logger = get_logger(__name__)


# ── Abstract Interface ────────────────────────────────────────────────────

class FacialKioskRepository(ABC):
    """
    Defines all persistence operations for the facial kiosk system.
    Implementations must honour these contracts regardless of the
    underlying store (SQLite, Postgres + pgvector, Pinecone, etc.).
    """

    # ── Users ─────────────────────────────────────────────────────────────

    @abstractmethod
    async def create_user(self, **fields: Any) -> User:
        """Create and persist a new User. Returns the saved instance."""
        ...

    @abstractmethod
    async def get_user_by_id(self, user_id: str) -> User | None:
        """Return a User by primary key, or None if not found."""
        ...

    @abstractmethod
    async def delete_user(self, user_id: str) -> bool:
        """Hard-delete a user and all cascaded records. Returns True if deleted."""
        ...

    # ── Face Records ──────────────────────────────────────────────────────

    @abstractmethod
    async def upsert_face_record(
        self,
        user_id: str,
        image_path: str,
        embedding: list[float],
        model_name: str,
        detector_name: str,
    ) -> FaceRecord:
        """
        Store a face record for a user.
        One record per user — replaces any existing record.
        """
        ...

    @abstractmethod
    async def get_face_record(self, user_id: str) -> FaceRecord | None:
        """Return the FaceRecord for a user, or None."""
        ...

    @abstractmethod
    async def find_closest_match(
        self,
        probe_embedding: list[float],
        threshold: float,
    ) -> tuple[User, float] | None:
        """
        Search all stored embeddings for the closest match to probe_embedding.
        Returns (User, distance) if distance <= threshold, else None.

        SQLite implementation uses cosine distance via NumPy.
        Vector DB implementation overrides with ANN query.
        """
        ...

    @abstractmethod
    async def get_all_face_records(self) -> list[FaceRecord]:
        """Return all stored face records (used for brute-force search)."""
        ...

    # ── Audit ─────────────────────────────────────────────────────────────

    @abstractmethod
    async def create_audit_log(
        self,
        request_id: str,
        action: str,
        status: str,
        client_ip: str,
        execution_ms: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> AuditLog:
        ...

    @abstractmethod
    async def get_audit_logs(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AuditLog]:
        ...

    # ── Verification Logs ─────────────────────────────────────────────────

    @abstractmethod
    async def create_verification_log(
        self,
        request_id: str,
        verified: bool,
        user_id: str | None = None,
        distance: float | None = None,
        threshold: float | None = None,
        liveness_score: float | None = None,
    ) -> VerificationLog:
        ...

    # ── Cleanup ───────────────────────────────────────────────────────────

    @abstractmethod
    async def delete_records_older_than(self, days: int) -> int:
        """Delete users (and cascaded records) older than N days. Returns count."""
        ...


# ── SQLite Implementation ─────────────────────────────────────────────────

class SQLiteRepository(FacialKioskRepository):
    """
    Concrete repository backed by SQLite via async SQLAlchemy.
    Embeddings stored as JSON float lists — cosine distance computed in NumPy.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ── Users ─────────────────────────────────────────────────────────────

    async def create_user(self, **fields: Any) -> User:
        user = User(**fields)
        self._session.add(user)
        await self._session.flush()
        logger.debug("User created", user_id=user.id)
        return user

    async def get_user_by_id(self, user_id: str) -> User | None:
        result = await self._session.execute(
            select(User).where(User.id == user_id)
        )
        return result.scalar_one_or_none()

    async def delete_user(self, user_id: str) -> bool:
        user = await self.get_user_by_id(user_id)
        if not user:
            return False
        await self._session.delete(user)
        return True

    # ── Face Records ──────────────────────────────────────────────────────

    async def upsert_face_record(
        self,
        user_id: str,
        image_path: str,
        embedding: list[float],
        model_name: str,
        detector_name: str,
    ) -> FaceRecord:
        # Delete existing record if present (one per user policy)
        await self._session.execute(
            delete(FaceRecord).where(FaceRecord.user_id == user_id)
        )

        record = FaceRecord(
            user_id=user_id,
            image_path=image_path,
            embedding=json.dumps(embedding),
            model_name=model_name,
            detector_name=detector_name,
        )
        self._session.add(record)
        await self._session.flush()
        logger.debug("FaceRecord upserted", user_id=user_id, model=model_name)
        return record

    async def get_face_record(self, user_id: str) -> FaceRecord | None:
        result = await self._session.execute(
            select(FaceRecord).where(FaceRecord.user_id == user_id)
        )
        return result.scalar_one_or_none()

    async def get_all_face_records(self) -> list[FaceRecord]:
        result = await self._session.execute(select(FaceRecord))
        return list(result.scalars().all())

    async def find_closest_match(
        self,
        probe_embedding: list[float],
        threshold: float,
    ) -> tuple[User, float] | None:
        """
        Brute-force cosine distance search across all stored embeddings.
        Acceptable for small datasets (kiosk use case).
        Override with ANN query when switching to a vector DB.
        """
        records = await self.get_all_face_records()
        if not records:
            return None

        probe  = np.array(probe_embedding, dtype=np.float32)
        probe /= np.linalg.norm(probe) + 1e-10   # normalise

        best_distance = float("inf")
        best_record:  FaceRecord | None = None

        for record in records:
            stored  = np.array(json.loads(record.embedding), dtype=np.float32)
            stored /= np.linalg.norm(stored) + 1e-10
            distance = float(1.0 - np.dot(probe, stored))  # cosine distance

            if distance < best_distance:
                best_distance = distance
                best_record   = record

        if best_record is None or best_distance > threshold:
            return None

        user = await self.get_user_by_id(best_record.user_id)
        if user is None:
            return None

        return (user, best_distance)

    # ── Audit ─────────────────────────────────────────────────────────────

    async def create_audit_log(
        self,
        request_id: str,
        action: str,
        status: str,
        client_ip: str,
        execution_ms: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> AuditLog:
        log = AuditLog(
            request_id=request_id,
            action=action,
            status=status,
            client_ip=client_ip,
            execution_ms=execution_ms,
            details=json.dumps(details) if details else None,
        )
        self._session.add(log)
        await self._session.flush()
        return log

    async def get_audit_logs(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AuditLog]:
        result = await self._session.execute(
            select(AuditLog)
            .order_by(AuditLog.timestamp.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    # ── Verification Logs ─────────────────────────────────────────────────

    async def create_verification_log(
        self,
        request_id: str,
        verified: bool,
        user_id: str | None = None,
        distance: float | None = None,
        threshold: float | None = None,
        liveness_score: float | None = None,
    ) -> VerificationLog:
        log = VerificationLog(
            request_id=request_id,
            user_id=user_id,
            distance=distance,
            threshold=threshold,
            verified=verified,
            liveness_score=liveness_score,
        )
        self._session.add(log)
        await self._session.flush()
        return log

    # ── Cleanup ───────────────────────────────────────────────────────────

    async def delete_records_older_than(self, days: int) -> int:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        result = await self._session.execute(
            select(User).where(User.created_at < cutoff)
        )
        users = list(result.scalars().all())

        for user in users:
            await self._session.delete(user)

        count = len(users)
        if count:
            logger.info("Cleanup: deleted old records", count=count, cutoff_days=days)
        return count


# ── Dependency factory ────────────────────────────────────────────────────

def get_repository(session: AsyncSession) -> FacialKioskRepository:
    """
    Returns the active repository implementation.
    To swap to a vector DB: change this to return VectorDBRepository(session).
    Everything else in the app stays unchanged.
    """
    return SQLiteRepository(session)
