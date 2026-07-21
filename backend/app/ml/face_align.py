"""
app/ml/face_align.py

5-point similarity-transform face alignment, the standard preprocessing
step for ArcFace-family recognition models (w600k_mbf included).

Given the 5 facial keypoints SCRFD predicts (left eye, right eye, nose,
left mouth corner, right mouth corner, in that order) this warps the face
into a canonical 112x112 crop matching the layout the recognition model
was trained on.
"""
import cv2
import numpy as np

# Canonical 112x112 ArcFace landmark template.
# Order: left eye, right eye, nose, left mouth corner, right mouth corner.
ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def norm_crop(img: np.ndarray, landmarks: np.ndarray, image_size: int = 112) -> np.ndarray:
    """
    Warp `img` so the given 5 landmarks align to the canonical ArcFace
    template, producing an `image_size` x `image_size` crop.

    landmarks: (5, 2) float array of [x, y] points in the *original* image's
    coordinate space, in the same left-eye/right-eye/nose/mouth-l/mouth-r
    order as ARCFACE_DST.
    """
    dst = ARCFACE_DST
    if image_size != 112:
        dst = dst * (image_size / 112.0)

    landmarks = np.asarray(landmarks, dtype=np.float32)

    # estimateAffinePartial2D fits a similarity transform (scale + rotation +
    # translation) from src -> dst, which is what insightface's own
    # alignment uses under the hood (there it's done via
    # skimage.transform.SimilarityTransform; this is the OpenCV-only
    # equivalent, avoiding an extra dependency).
    matrix, _ = cv2.estimateAffinePartial2D(landmarks, dst, method=cv2.LMEDS)

    if matrix is None:
        # Degenerate landmarks (shouldn't normally happen) — fall back to
        # a simple resize of the whole image rather than raising, so a
        # single bad frame doesn't crash the pipeline.
        return cv2.resize(img, (image_size, image_size))

    warped = cv2.warpAffine(img, matrix, (image_size, image_size), borderValue=0.0)
    return warped
