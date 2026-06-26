import numpy as np

from app.core.config import settings
from app.services.classification.model_loader import predict


def normalize_name(name: str) -> str:
    return name.strip().lower().replace("_", " ").replace("-", " ")


def run_pill_check(image_rgb: np.ndarray, claimed_drug: str, top_k: int = 3) -> dict:
    top_predictions = predict(image_rgb, top_k=top_k)
    predicted_drug, confidence = top_predictions[0]

    claimed_norm = normalize_name(claimed_drug)
    predicted_norm = normalize_name(predicted_drug)

    if confidence < settings.MATCH_CONFIDENCE_THRESHOLD:
        verdict = "uncertain"
    elif claimed_norm == predicted_norm:
        verdict = "match"
    else:
        verdict = "mismatch"

    return {
        "predicted_drug": predicted_drug,
        "confidence": confidence,
        "verdict": verdict,
        "top_predictions": [{"label": label, "confidence": conf} for label, conf in top_predictions],
    }
