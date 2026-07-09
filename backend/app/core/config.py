"""
app/core/config.py
All application settings loaded from environment variables or .env file.
Override any value by setting the corresponding env var.
"""
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── App ────────────────────────────────────────────────────────────────
    APP_NAME: str = "Facial Identity Management System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"

    # ── Database ───────────────────────────────────────────────────────────
    DATABASE_URL: str = "sqlite+aiosqlite:///./facial_kiosk.db"

    # ── Storage ────────────────────────────────────────────────────────────
    STORAGE_DIR: Path = Path("app/storage/faces")
    MAX_STORED_FACES_PER_USER: int = 1  # enforced at repository layer

    # ── Security / JWT ────────────────────────────────────────────────────
    JWT_SECRET: str = "change-me-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRY_MINUTES: int = 15

    # Admin key — passed as Authorization: Bearer {key} on admin endpoints
    ADMIN_API_KEY: str = "change-admin-key-in-production"

    # Signed challenge TTL (seconds) — nonce expires after this window
    CHALLENGE_TTL_SECONDS: int = 30

    # ── CORS ──────────────────────────────────────────────────────────────
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    # ── Rate limiting ─────────────────────────────────────────────────────
    RATE_LIMIT_REGISTER: str = "5/minute"
    RATE_LIMIT_VERIFY: str = "10/minute"

    # ── Image validation ──────────────────────────────────────────────────
    MAX_IMAGE_SIZE_BYTES: int = 5 * 1024 * 1024   # 5 MB
    MIN_IMAGE_WIDTH: int = 320
    MIN_IMAGE_HEIGHT: int = 240
    MAX_IMAGE_WIDTH: int = 4096
    MAX_IMAGE_HEIGHT: int = 4096
    MIN_SHARPNESS_SCORE: float = 100.0             # Laplacian variance threshold
    ALLOWED_MIME_TYPES: list[str] = ["image/jpeg", "image/png", "image/webp"]

    # ── Face detection ────────────────────────────────────────────────────
    RETINAFACE_DETECTOR: str = "retinaface"
    DEEPFACE_MODEL: str = "Facenet512"
    DEEPFACE_DISTANCE_METRIC: str = "cosine"
    VERIFICATION_THRESHOLD: float = 0.40

    # ── Liveness ──────────────────────────────────────────────────────────
    LIVENESS_THRESHOLD: float = 0.5               # MiniFASNet score threshold
    MINIFAS_MODEL_DIR: str = "app/services/minifas_weights"

    # ── Cleanup ───────────────────────────────────────────────────────────
    CLEANUP_AGE_DAYS: int = 30
    CLEANUP_INTERVAL_HOURS: int = 24              # background task interval


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance — import this everywhere."""
    return Settings()
