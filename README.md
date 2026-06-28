# 💊 Pill Identity Checker

A Streamlit web app that identifies pills from photos across **112 known pill types** using a self-trained EfficientNet-B0 classifier. Optionally flags a mismatch between the predicted identity and a claimed drug name.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red)
![Accuracy](https://img.shields.io/badge/Top--1_Accuracy-97.2%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

**[🚀 Live Demo → fake-medicine-detection-zrik.streamlit.app](https://fake-medicine-detection-zrik.streamlit.app/)**

---

## ✨ Features

- 📸 Upload any pill photo (JPG / PNG)
- 🔍 Identifies the pill across 112 known classes with confidence scores
- ✅ / 🚨 Match or mismatch verdict against a claimed drug name
- 📊 Top-3 predictions with confidence bars
- ⚡ Runs on CPU — no GPU needed

---

## 🧠 Scope Note

There is no public dataset of real-vs-counterfeit medicine images. Instead, this project trains a **pill identity classifier** on the [OGYEIv2](https://www.kaggle.com/datasets/richardradli/ogyeiv2) dataset (112 genuine pill classes) and flags a visual mismatch between the photographed pill and the user's stated drug name — a legitimate proxy signal for verification, but **not** literal chemical counterfeit detection.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| [Streamlit](https://streamlit.io) | Web app framework |
| EfficientNet-B0 (ONNX) | Pill classification model |
| [ONNX Runtime](https://onnxruntime.ai) | CPU inference |
| [Pillow](https://python-pillow.org) | Image preprocessing |
| PyTorch + timm | Training (Kaggle GPU) |

---

## 📈 Model Performance

| Metric | Value |
|--------|-------|
| Top-1 Accuracy | **97.2%** |
| Top-3 Accuracy | **99.85%** |
| Classes | 112 pill types |
| Dataset | OGYEIv2 (Kaggle) |
| Inference | ONNX Runtime (CPU) |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/krishrakholiya32/fake-medicine-detection.git
cd fake-medicine-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ⚙️ How It Works

1. Upload a pill photo — the app resizes it to 224×224 using PIL
2. The ONNX model runs inference on CPU (~50ms)
3. Top-3 predicted classes and confidence scores are returned
4. If a claimed drug name is provided, the top prediction is compared against it:
   - **Match** — predicted identity matches the claimed name (confidence ≥ 50%)
   - **Mismatch** — predicted identity does NOT match (confidence ≥ 50%)
   - **Uncertain** — confidence below 50%, cannot reliably verify

---

## 📝 License

MIT — feel free to use, modify, and share.
