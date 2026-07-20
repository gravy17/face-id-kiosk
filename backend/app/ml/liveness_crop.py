"""
app/ml/liveness_crop.py

Re-implementation of the bounding-box expansion/crop used by
Silent-Face-Anti-Spoofing to prepare a face crop for MiniFASNet: the
detected bbox is expanded by `scale` around its center (clamped to image
bounds), then resized to the model's input size.
"""
import cv2
import numpy as np


def _expanded_box(
    src_w: int,
    src_h: int,
    bbox: tuple[int, int, int, int],
    scale: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    box_w, box_h = x2 - x1, y2 - y1

    scale = min((src_h - 1) / box_h, min((src_w - 1) / box_w, scale))

    new_w, new_h = box_w * scale, box_h * scale
    center_x, center_y = x1 + box_w / 2, y1 + box_h / 2

    left, top = center_x - new_w / 2, center_y - new_h / 2
    right, bottom = center_x + new_w / 2, center_y + new_h / 2

    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > src_w - 1:
        left -= right - (src_w - 1)
        right = src_w - 1
    if bottom > src_h - 1:
        top -= bottom - (src_h - 1)
        bottom = src_h - 1

    return int(max(0, left)), int(max(0, top)), int(min(src_w - 1, right)), int(min(src_h - 1, bottom))


def crop_for_liveness(
    img: np.ndarray,
    bbox: tuple[int, int, int, int],
    scale: float,
    out_size: int,
) -> np.ndarray:
    """
    img: full BGR image.
    bbox: (x1, y1, x2, y2) detected face box, in img's coordinate space.
    scale: expansion factor around the box center before resizing.
    out_size: side length of the returned square crop.
    """
    h, w = img.shape[:2]
    left, top, right, bottom = _expanded_box(w, h, bbox, scale)
    cropped = img[top : bottom + 1, left : right + 1]
    if cropped.size == 0:
        cropped = img
    return cv2.resize(cropped, (out_size, out_size))
