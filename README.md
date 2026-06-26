# Fake Medicine Detection

A full-stack web application that verifies whether a pill's visual appearance matches what it claims to be. Upload a photo of a pill, enter the expected drug name, and get an instant match/mismatch/uncertain verdict powered by a self-trained EfficientNet-B0 classifier running as ONNX on CPU.

## Scope note

There is no public dataset of labeled real-vs-counterfeit medicine images (collecting real counterfeits is dangerous and illegal). This project instead trains a **pill identity classifier** on the [OGYEIv2](https://www.kaggle.com/datasets/richardradli/ogyeiv2) dataset (112 genuine pill classes) and flags a mismatch between the photographed pill's predicted identity and the user's stated identity as a proxy signal — a legitimate and defensible framing, but not literal counterfeit detection.

## Features

- Photo upload (JPEG / PNG / WebP)
- Verdict: **Match**, **Mismatch**, or **Uncertain** (below confidence threshold)
- Top-3 predicted classes with confidence scores
- Full check history with pagination
- Dockerized stack — one command to run

## Tech stack

| Layer | Technology |
|---|---|
| Model | EfficientNet-B0 (timm), exported to ONNX |
| Backend | FastAPI, SQLAlchemy async, PostgreSQL, ONNX Runtime |
| Frontend | React 18, Vite, TypeScript, Tailwind CSS |
| Infra | Docker Compose, Nginx reverse proxy |
| Training | PyTorch, Kaggle Notebooks (GPU) |

## Architecture

```
browser
  └── nginx :8081
        ├── /api/*   → FastAPI backend :8000
        └── /*       → React SPA :80

backend
  ├── POST /api/check/pill   — classify pill image, return verdict
  ├── GET  /api/history      — list past checks
  └── GET  /health

models/
  ├── pill_classifier_effnetb0.onnx   (gitignored — see Training)
  └── labels.json                      (gitignored — see Training)
```

## Quick start (Docker)

### Prerequisites

- Docker Desktop (or Docker + Docker Compose v2)
- Trained model files in `models/` (see [Training](#training) or skip to run without a model)

```bash
git clone https://github.com/krishrakholiya32/fake-medicine-detection.git
cd fake-medicine-detection

# Copy .env.example and adjust if needed
cp .env.example .env

# Place your model files (pill_classifier_effnetb0.onnx + labels.json) in models/
# or run: python scripts/setup_models.py  (if they're in the project root)

docker-compose up --build
```

Open **http://localhost:8081** in your browser.

| Port | Service |
|---|---|
| 8081 | Main app (nginx) |
| 8002 | Backend API directly |
| 3002 | Frontend directly |

### Without the trained model

The stack starts without the model files — the backend will return a 500 when you try to classify a pill. Train the model first (see below) or obtain `pill_classifier_effnetb0.onnx` + `labels.json` from a training run.

## Training

Training runs on Kaggle (free GPU). All scripts are in `training/scripts/`.

### 1. Set up a Kaggle Notebook

Create a new notebook and add the dataset:
- **Dataset**: [`richardradli/ogyeiv2`](https://www.kaggle.com/datasets/richardradli/ogyeiv2)

Upload the contents of `training/` to the notebook (or clone the repo there).

### 2. Prepare the dataset

```bash
python training/scripts/prepare_dataset.py \
  --raw_dir /kaggle/input/datasets/richardradli/ogyeiv2/ogyeiv2/ogyeiv2 \
  --output_dir /kaggle/working/data
```

This reorganizes the dataset into `{train,val,test}/<class_name>/*.jpg` and writes `labels.json`.

**Dataset layout note**: actual layout is `<raw_dir>/ogyeiv2/ogyeiv2/{train,valid,test}/images/<class>_<s|u>_<index>.jpg` — the script parses class names from filenames and uses the dataset's own split.

### 3. Train

```bash
python training/scripts/train.py --config training/configs/train_config.yaml
```

Two-phase fine-tuning:
1. Freeze all but the classification head, train for warmup epochs
2. Unfreeze last 3 EfficientNet blocks, train with lower LR

Expected results: **~97% top-1 accuracy, ~99.8% top-3 accuracy** on the 672-image test set (112 classes).

### 4. Export to ONNX

```bash
python training/scripts/export_model.py
```

Exports to ONNX (TorchScript exporter, `dynamo=False`) and verifies numerical agreement between PyTorch and ONNX outputs in probability space (tolerance 0.01).

### 5. Copy model to project

Download `pill_classifier_effnetb0.onnx` and `labels.json` from the Kaggle output, place them in the project root, then:

```bash
python scripts/setup_models.py
```

This copies them into `models/` where Docker mounts them.

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/check/pill` | Classify a pill image. Form fields: `file` (image), `claimed_drug` (string) |
| `GET` | `/api/history` | List all past checks (newest first) |
| `GET` | `/api/history/{check_id}` | Single check by ID |
| `GET` | `/health` | Health check |

### Example

```bash
curl -X POST http://localhost:8002/api/check/pill \
  -F "file=@pill.jpg" \
  -F "claimed_drug=advil_ultra_forte"
```

```json
{
  "id": 1,
  "claimed_drug": "advil_ultra_forte",
  "predicted_drug": "advil_ultra_forte",
  "confidence": 0.9982,
  "verdict": "match",
  "top_predictions": [
    {"label": "advil_ultra_forte", "confidence": 0.9982},
    {"label": "brufen_400_mg",     "confidence": 0.0009},
    {"label": "nurofen_200_mg",    "confidence": 0.0004}
  ]
}
```

**Verdict logic**:
- `match` — predicted class matches claimed drug AND confidence ≥ threshold (default 0.5)
- `mismatch` — predicted class does NOT match claimed drug AND confidence ≥ threshold
- `uncertain` — confidence below threshold (model is not sure)

## Configuration

Copy `.env.example` to `.env` and adjust:

| Variable | Default | Description |
|---|---|---|
| `MATCH_CONFIDENCE_THRESHOLD` | `0.5` | Minimum confidence to give a match/mismatch verdict |
| `BACKEND_CPU_LIMIT` | `1` | Docker CPU quota for the backend container |
| `BACKEND_MEM_LIMIT` | `2G` | Docker memory limit for the backend container |

## Deployment

See [DEPLOY.md](DEPLOY.md) for step-by-step instructions to deploy on Oracle Cloud Always Free (ARM A1 VM).

## License

MIT — see [LICENSE](LICENSE).
