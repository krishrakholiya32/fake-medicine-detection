"""Loads the exported ONNX pill classifier + label list once and reuses them."""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime as ort

from app.core.config import settings

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_SIZE = 224


@lru_cache(maxsize=1)
def get_session() -> ort.InferenceSession:
    model_path = Path(settings.MODELS_PATH) / settings.MODEL_FILENAME
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])


@lru_cache(maxsize=1)
def get_labels() -> list[str]:
    labels_path = Path(settings.MODELS_PATH) / settings.LABELS_FILENAME
    with open(labels_path) as f:
        return json.load(f)


def preprocess(image_rgb: np.ndarray) -> np.ndarray:
    import cv2

    # INTER_AREA (not INTER_LINEAR) for downsampling — real photos are often much
    # larger than 224x224, and plain bilinear resize aliases badly at large shrink
    # factors compared to the anti-aliased resize PIL/torchvision used during training.
    resized = cv2.resize(image_rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    img = resized.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = img.transpose(2, 0, 1)
    return img[np.newaxis, ...].astype(np.float32)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def predict(image_rgb: np.ndarray, top_k: int = 3) -> list[tuple[str, float]]:
    session = get_session()
    labels = get_labels()
    inputs = preprocess(image_rgb)
    logits = session.run(["logits"], {"input": inputs})[0][0]
    probs = softmax(logits)
    top_indices = np.argsort(probs)[::-1][:top_k]
    return [(labels[i], float(probs[i])) for i in top_indices]
