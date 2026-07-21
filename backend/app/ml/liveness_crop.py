"""
app/ml/liveness_crop.py

Face-region crop for MiniFASNet, matching the reference implementation in
yakhyo/face-anti-spoofing's onnx_inference.py (the source of the
MiniFASNetV2.onnx weights this app uses): the detected bbox is expanded by
`scale` around its center, then clamped (not shifted) to image bounds, then
resized to the model's input size.
"""
import cv2
import numpy as np


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
    src_h, src_w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    box_w, box_h = x2 - x1, y2 - y1

    # min(...) so the expanded box never exceeds the image bounds outright
    # (mirrors the reference implementation's own scale clamp).
    eff_scale = min((src_h - 1) / box_h, (src_w - 1) / box_w, scale)

    new_w = box_w * eff_scale
    new_h = box_h * eff_scale
    center_x = x1 + box_w / 2
    center_y = y1 + box_h / 2

    left   = max(0, int(center_x - new_w / 2))
    top    = max(0, int(center_y - new_h / 2))
    right  = min(src_w - 1, int(center_x + new_w / 2))
    bottom = min(src_h - 1, int(center_y + new_h / 2))

    cropped = img[top : bottom + 1, left : right + 1]
    if cropped.size == 0:
        cropped = img
    return cv2.resize(cropped, (out_size, out_size))
