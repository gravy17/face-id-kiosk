"""
app/services/image_service.py
Handles saving and deleting face images from disk storage.
Keeps storage organised by user ID subdirectory.
"""
import uuid
from pathlib import Path

import cv2
import numpy as np

from app.core.config import Settings, get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)


class ImageService:

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings   = settings or get_settings()
        self._storage    = self._settings.STORAGE_DIR
        self._storage.mkdir(parents=True, exist_ok=True)

    def save_face(self, user_id: str, image: np.ndarray) -> str:
        """
        Save a face image to storage/faces/{user_id}/{uuid}.jpg.
        Returns the relative path string for storage in the database.
        """
        user_dir = self._storage / user_id
        user_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{uuid.uuid4().hex}.jpg"
        filepath = user_dir / filename

        success = cv2.imwrite(str(filepath), image)
        if not success:
            raise RuntimeError(f"Failed to write face image to {filepath}")

        relative_path = str(filepath)
        logger.debug("Face image saved", path=relative_path, user_id=user_id)
        return relative_path

    def delete_user_images(self, user_id: str) -> int:
        """
        Delete all images for a user. Returns the count of deleted files.
        Called when a user is deleted or their face record is replaced.
        """
        user_dir = self._storage / user_id
        if not user_dir.exists():
            return 0

        count = 0
        for f in user_dir.iterdir():
            if f.is_file():
                f.unlink()
                count += 1

        try:
            user_dir.rmdir()
        except OSError:
            pass   # not empty — that's fine

        logger.debug("User images deleted", user_id=user_id, count=count)
        return count

    def delete_image(self, image_path: str) -> bool:
        """Delete a single image by path. Returns True if deleted."""
        p = Path(image_path)
        if p.exists() and p.is_file():
            p.unlink()
            logger.debug("Image deleted", path=image_path)
            return True
        return False


def get_image_service() -> ImageService:
    return ImageService()
