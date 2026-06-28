"""Pill Identity Checker — Streamlit app.

Identifies pill photos across 112 known pill types (OGYEIv2 dataset)
using a self-trained EfficientNet-B0 ONNX classifier. Optionally
flags a mismatch between the predicted identity and a claimed drug name.

Scope: pill appearance identity verification — not chemical counterfeit detection.
"""

import json
import numpy as np
import streamlit as st
from pathlib import Path
from PIL import Image

MODEL_PATH = Path("models/pill_classifier_effnetb0.onnx")
LABELS_PATH = Path("models/labels.json")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_SIZE = 224
MATCH_THRESHOLD = 0.5


@st.cache_resource(show_spinner=False)
def _load_session():
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(MODEL_PATH), sess_options=opts, providers=["CPUExecutionProvider"])


@st.cache_resource(show_spinner=False)
def _load_labels() -> list[str]:
    with open(LABELS_PATH) as f:
        return json.load(f)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def predict(image_rgb: np.ndarray, top_k: int = 3) -> list[tuple[str, float]]:
    """Return top-k (label, probability) pairs."""
    session = _load_session()
    labels = _load_labels()
    resized = np.array(Image.fromarray(image_rgb).resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS))
    img = resized.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = img.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
    logits = session.run(["logits"], {"input": img})[0][0]
    probs = _softmax(logits)
    top_idx = np.argsort(probs)[::-1][:top_k]
    return [(labels[i], float(probs[i])) for i in top_idx]


def _fmt(label: str) -> str:
    return label.replace("_", " ").title()


def _is_match(predicted: str, claimed: str) -> bool:
    p = predicted.lower()
    c = claimed.strip().lower().replace(" ", "_").replace("-", "_")
    return c in p or p in c


# ── UI ──────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Pill Checker", page_icon="💊", layout="centered")

st.title("💊 Pill Identity Checker")
st.markdown(
    "Upload a pill photo to identify it across **112 known pill types**. "
    "Optionally enter the claimed drug name to check for a mismatch."
)

with st.expander("ℹ️ About this model & limitations"):
    st.markdown(
        "- Self-trained **EfficientNet-B0** on OGYEIv2 dataset (112 pill classes)\n"
        "- **Top-1 accuracy 97.2% · Top-3 accuracy 99.85%** on held-out test set\n"
        "- Exported to ONNX · Preprocessing: INTER_AREA resize to match training (PIL anti-alias)\n\n"
        "**Important limitations:**\n"
        "- Identifies pill *appearance* only — cannot detect chemical counterfeits or contamination\n"
        "- Only covers the 112 pill classes in the OGYEIv2 dataset\n"
        "- For any medical decision, consult a pharmacist or physician"
    )

col1, col2 = st.columns([3, 2])
with col1:
    uploaded = st.file_uploader("Upload a pill photo", type=["jpg", "jpeg", "png"])
with col2:
    claimed = st.text_input("Claimed drug name (optional)", placeholder="e.g. aspirin 500mg")

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    image_rgb = np.array(image)

    with st.spinner("Identifying pill…"):
        predictions = predict(image_rgb)

    st.image(image, caption="Uploaded pill photo", width=360)
    st.divider()

    top_name, top_conf = predictions[0]
    top_label = _fmt(top_name)

    if claimed.strip():
        if top_conf < MATCH_THRESHOLD:
            st.warning(f"### ⚠️ UNCERTAIN — {top_conf:.1%} confidence")
            st.markdown(
                f"Best guess: **{top_label}** — but confidence is below the verification "
                f"threshold ({MATCH_THRESHOLD:.0%}). Cannot reliably confirm or deny the claimed identity."
            )
        elif _is_match(top_name, claimed):
            st.success(f"### ✅ MATCH — {top_conf:.1%} confidence")
            st.markdown(
                f"Predicted identity **{top_label}** matches the claimed name **{claimed.strip()}**."
            )
        else:
            st.error(f"### 🚨 MISMATCH — Predicted: {top_label}")
            st.markdown(
                f"Claimed: **{claimed.strip()}** · Predicted: **{top_label}** ({top_conf:.1%}). "
                "The pill's appearance does not match the claimed identity."
            )
    else:
        st.info(f"### 🔍 Identified: {top_label}")
        st.markdown(f"Confidence: **{top_conf:.1%}**")

    st.subheader("Top 3 predictions")
    for name, conf in predictions:
        label = _fmt(name)
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.progress(conf, text=label)
        with col_b:
            st.write(f"**{conf:.1%}**")

st.divider()
st.caption(
    "Trained on OGYEIv2 (112 pill classes) · "
    "EfficientNet-B0 + ONNX Runtime · "
    "[GitHub](https://github.com/krishrakholiya32/fake-medicine-detection)"
)
