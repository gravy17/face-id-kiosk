"""
app/ml/onnx_session.py

Small shared helper for creating onnxruntime InferenceSession objects.
Centralised so provider selection / error messages are consistent across
the detection, recognition and liveness services.
"""
from pathlib import Path

import onnxruntime as ort

from app.core.logger import get_logger

logger = get_logger(__name__)


def load_session(model_path: Path | str, providers: list[str]) -> ort.InferenceSession:
    """
    Load an onnxruntime InferenceSession from disk.

    providers: preferred execution providers, in priority order (e.g.
    ["CUDAExecutionProvider", "CPUExecutionProvider"]). Only providers that
    onnxruntime actually has installed support for are passed through;
    "CPUExecutionProvider" is always appended as a guaranteed fallback.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"ONNX model not found at '{model_path}'. "
            f"Place the model file there or point the relevant "
            f"*_MODEL_PATH setting at its actual location."
        )

    available = set(ort.get_available_providers())
    selected = [p for p in providers if p in available]
    if "CPUExecutionProvider" not in selected:
        selected.append("CPUExecutionProvider")

    session = ort.InferenceSession(str(model_path), providers=selected)

    logger.info(
        "ONNX model loaded",
        path=str(model_path),
        providers=selected,
    )
    return session
