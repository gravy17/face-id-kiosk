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

    # ── ONNX model files ───────────────────────────────────────────────────
    # Root directory containing detection/, recognition/, liveness/ subfolders
    MODELS_DIR: Path = Path("models")

    DETECTION_MODEL_PATH:   Path | None = None   # default: MODELS_DIR/detection/det_500m.onnx
    RECOGNITION_MODEL_PATH: Path | None = None   # default: MODELS_DIR/recognition/w600k_mbf.onnx
    LIVENESS_MODEL_PATH:    Path | None = None   # default: MODELS_DIR/liveness/MiniFASNetV2.onnx

    # onnxruntime execution providers, in priority order. CPU is always
    # appended automatically as a fallback if not already present.
    ONNX_PROVIDERS: list[str] = ["CPUExecutionProvider"]

    # ── Face detection (SCRFD) ──────────────────────────────────────────────
    DETECTION_INPUT_SIZE: int = 640          # square input fed to SCRFD (letterboxed)
    DETECTION_CONF_THRESHOLD: float = 0.5    # min detection score to keep a candidate
    DETECTION_NMS_THRESHOLD: float = 0.4     # IoU threshold for NMS

    # ── Face recognition (ArcFace / w600k_mbf) ───────────────────────────────
    RECOGNITION_INPUT_SIZE: int = 112
    # NOTE: this threshold was tuned for a different embedding model
    # (Facenet512 cosine distance). Cosine-distance geometry differs between
    # embedding models, so re-calibrate this empirically against a labelled
    # pair dataset before relying on it in production with w600k_mbf.
    VERIFICATION_THRESHOLD: float = 0.40

    # ── Liveness (MiniFASNetV2) ──────────────────────────────────────────────
    LIVENESS_THRESHOLD: float = 0.5          # min "real" softmax probability to pass
    LIVENESS_INPUT_SIZE: int = 80
    # Crop expansion factor around the detected face bbox before resizing to
    # LIVENESS_INPUT_SIZE. Confirmed 2.7 for MiniFASNetV2 per
    # https://github.com/yakhyo/face-anti-spoofing (source of this weight
    # file) — its onnx_inference.py hardcodes scale=2.7 for V2 (4.0 for
    # V1SE). Only change this if you swap in a different exported model.
    LIVENESS_CROP_SCALE: float = 2.7
    # Which softmax output index corresponds to "real" (vs spoof/unclear).
    # Confirmed 1 per yakhyo/face-anti-spoofing's onnx_inference.py:
    # `"Real" if label_idx == 1 else "Fake"`.
    LIVENESS_REAL_CLASS_INDEX: int = 1
    # Channel order the model expects. Confirmed "bgr" — the reference
    # onnx_inference.py never converts BGR->RGB, it feeds cv2's native
    # channel order straight through.
    LIVENESS_CHANNEL_ORDER: str = "bgr"      # "bgr" or "rgb"
    # Whether to scale pixel values to [0, 1] before inference. Confirmed
    # False — yakhyo/face-anti-spoofing's onnx_inference.py does NOT divide
    # by 255; it feeds raw float32 pixel values (0-255 range) directly.
    # This was the actual bug in the original implementation here: dividing
    # by 255 when the model expects raw pixel magnitudes produces
    # meaningless logits, which is why every real face scored as spoof.
    LIVENESS_NORMALIZE_255: bool = False

    # ── Cleanup ───────────────────────────────────────────────────────────
    CLEANUP_AGE_DAYS: int = 30
    CLEANUP_INTERVAL_HOURS: int = 24              # background task interval

    def detection_model_path(self) -> Path:
        return self.DETECTION_MODEL_PATH or (self.MODELS_DIR / "detection" / "det_500m.onnx")

    def recognition_model_path(self) -> Path:
        return self.RECOGNITION_MODEL_PATH or (self.MODELS_DIR / "recognition" / "w600k_mbf.onnx")

    def liveness_model_path(self) -> Path:
        return self.LIVENESS_MODEL_PATH or (self.MODELS_DIR / "liveness" / "MiniFASNetV2.onnx")


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance — import this everywhere."""
    return Settings()
