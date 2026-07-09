"""
app/services/report_service.py
Builds the structured detail dict written to AuditLog.details for each action.
Centralises what gets logged so it's consistent and easy to extend.
"""
from typing import Any

from app.services.deepface_service import EmbeddingResult, VerificationResult
from app.services.liveness_service import LivenessResult
from app.services.retinaface_service import DetectedFace


def registration_details(
    user_id: str,
    detected_face: DetectedFace,
    embedding_result: EmbeddingResult,
    liveness_result: LivenessResult,
    image_path: str,
) -> dict[str, Any]:
    return {
        "user_id":       user_id,
        "image_path":    image_path,
        "face": {
            "confidence":  round(detected_face.confidence, 4),
            "area":        detected_face.facial_area,
        },
        "embedding": {
            "model":      embedding_result.model_name,
            "dimensions": embedding_result.dimensions,
        },
        "liveness": {
            "passed": liveness_result.passed,
            "score":  liveness_result.score,
            "model":  liveness_result.model_name,
        },
    }


def verification_details(
    liveness_result: LivenessResult,
    verification_result: VerificationResult | None,
    matched_user_id: str | None,
) -> dict[str, Any]:
    details: dict[str, Any] = {
        "liveness": {
            "passed": liveness_result.passed,
            "score":  liveness_result.score,
            "model":  liveness_result.model_name,
        },
    }

    if verification_result:
        details["match"] = {
            "verified":       verification_result.verified,
            "distance":       verification_result.distance,
            "threshold":      verification_result.threshold,
            "confidence":     verification_result.confidence,
            "matched_user":   matched_user_id,
        }

    return details


def rejection_details(reason: str, **extra: Any) -> dict[str, Any]:
    return {"rejection_reason": reason, **extra}
