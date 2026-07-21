#!/usr/bin/env python3
"""
scripts/tune_liveness.py

Empirically determine the correct liveness preprocessing settings for your
.onnx export, instead of guessing.

For MiniFASNetV2 sourced from yakhyo/face-anti-spoofing specifically, these
are already confirmed (straight from that repo's own onnx_inference.py) and
set as this app's defaults — you shouldn't need this script unless you swap
in a model from a different source:
    LIVENESS_CROP_SCALE=2.7
    LIVENESS_CHANNEL_ORDER=bgr
    LIVENESS_REAL_CLASS_INDEX=1
    LIVENESS_NORMALIZE_255=false   (raw 0-255 float32 pixels, NOT scaled)

Usage:
    python scripts/tune_liveness.py path/to/a_real_face_photo.jpg

Feed it a photo of an actual live face (a normal selfie is fine — doesn't
need to come through the app). The script:
  1. Reports the model's actual declared input shape (authoritative — compare
     against your LIVENESS_INPUT_SIZE setting).
  2. Runs your configured face detector to get a real bounding box.
  3. Sweeps crop scale x channel order x normalization, printing the full
     3-class softmax output for each combination.

How to read the results:
  - A real photo should score high on whichever class is "real" — and that
    score should peak sharply at the *correct* combination and be
    noticeably lower at the others. If nothing gets above ~0.5 no matter
    what you try, the mismatch is likely something other than these
    settings (e.g. per-channel mean/std normalisation instead of plain
    scaling, or this .onnx file isn't actually what you think it is).
  - Whichever class index is consistently high across a real photo is your
    LIVENESS_REAL_CLASS_INDEX.
  - Once you've identified the winning combination from a few different
    real photos (ideally shot with the actual kiosk webcam, in the actual
    kiosk lighting), set them in your .env:
        LIVENESS_CROP_SCALE=<winning scale>
        LIVENESS_CHANNEL_ORDER=<bgr|rgb>
        LIVENESS_NORMALIZE_255=<true|false>
        LIVENESS_REAL_CLASS_INDEX=<winning index>
  - Only after that's dialed in should you tune LIVENESS_THRESHOLD — pick a
    value comfortably between your real-face scores and your spoof-attempt
    scores (test with a photo-of-a-photo / phone screen too, if you can).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from app.core.config import get_settings
from app.ml.liveness_crop import crop_for_liveness
from app.ml.onnx_session import load_session

CANDIDATE_SCALES = [1.0, 1.5, 2.0, 2.7, 3.0, 4.0]
CANDIDATE_CHANNEL_ORDERS = ["bgr", "rgb"]
CANDIDATE_NORMALIZE = [False, True]


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} path/to/real_face_photo.jpg")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"File not found: {image_path}")
        sys.exit(1)

    settings = get_settings()
    img = cv2.imread(str(image_path))
    if img is None:
        print("Could not decode image — is it a valid JPEG/PNG?")
        sys.exit(1)

    # ── 1. Report the model's actual declared input shape ──────────────────
    liveness_path = settings.liveness_model_path()
    if not liveness_path.exists():
        print(f"Liveness model not found at {liveness_path}. "
              f"Set LIVENESS_MODEL_PATH or place it there.")
        sys.exit(1)

    session = load_session(liveness_path, settings.ONNX_PROVIDERS)
    declared_shape = session.get_inputs()[0].shape
    output_shape = session.get_outputs()[0].shape
    print(f"Model declared input shape:  {declared_shape}")
    print(f"Model declared output shape: {output_shape}  "
          f"(number of classes = last dim, should be 3 for MiniFASNet)")
    print(f"Your configured LIVENESS_INPUT_SIZE: {settings.LIVENESS_INPUT_SIZE}")
    print()

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # ── 2. Detect a real face to get a real bounding box ────────────────────
    from app.services.retinaface_service import get_face_detector
    detector = get_face_detector()
    try:
        detected = detector.detect_single_face(img)
    except Exception as exc:
        print(f"Detection failed on this photo: {exc}")
        print("Use a clearer single-face photo.")
        sys.exit(1)

    bbox = (
        detected.facial_area["x"],
        detected.facial_area["y"],
        detected.facial_area["x"] + detected.facial_area["w"],
        detected.facial_area["y"] + detected.facial_area["h"],
    )
    print(f"Detected face bbox: {bbox} (confidence {detected.confidence:.3f})")
    print()

    # ── 3. Sweep scale x channel order x normalization ───────────────────────
    header = f"{'scale':>6} | {'order':>5} | {'norm':>5} | {'class 0':>8} | {'class 1':>8} | {'class 2':>8}"
    print(header)
    print("-" * len(header))

    for scale in CANDIDATE_SCALES:
        crop = crop_for_liveness(img, bbox, scale=scale, out_size=settings.LIVENESS_INPUT_SIZE)

        for order in CANDIDATE_CHANNEL_ORDERS:
            frame = crop[:, :, ::-1] if order == "rgb" else crop

            for normalize in CANDIDATE_NORMALIZE:
                blob = frame.astype(np.float32)
                if normalize:
                    blob = blob / 255.0
                blob = np.ascontiguousarray(np.transpose(blob, (2, 0, 1))[None, ...])

                logits = session.run([output_name], {input_name: blob})[0]
                probs = softmax(logits.reshape(-1))

                cells = " | ".join(f"{p:8.4f}" for p in probs.tolist())
                print(f"{scale:>6} | {order:>5} | {str(normalize):>5} | {cells}")

    print()
    print("Look for the row with the highest score concentrated in one class —")
    print("that combination is almost certainly correct, and that class index")
    print("is your LIVENESS_REAL_CLASS_INDEX.")


if __name__ == "__main__":
    main()
