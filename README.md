# Ml Recommendation Engine Hybrid

An enterprise-grade solution engineered for high performance.

![Language](https://img.shields.io/badge/Language-Python-blue)
![Status](https://img.shields.io/badge/Status-Active-success)
![License](https://img.shields.io/badge/License-VisionQuantech%20Custom-orange)

## 🚀 Overview

Welcome to the **Ml Recommendation Engine Hybrid** repository. This project is built to deliver a robust and scalable solution tailored to modern development standards.

This repository implements a **hybrid recommendation system** in PyTorch that fuses **collaborative filtering** (matrix factorization + neural collaborative filtering) with **content-based signals**, served via a **FastAPI** inference layer and containerized with Docker.

## ✨ Features

- **High Performance:** Optimized for speed and efficiency.
- **Scalable Architecture:** Designed to grow with your needs.
- **Clean Codebase:** Follows best practices and industry standards.
- **Secure by Default:** Engineered with security in mind.
- **Four Model Architectures:** Matrix Factorization, Neural Collaborative Filtering, Two-Tower retrieval, and a Hybrid fusion model.
- **Training Pipeline:** Reusable `Trainer` class with MSE loss, Adam optimizer, GPU auto-detection, and RMSE/MAE evaluation.
- **REST API Serving:** FastAPI endpoints for health checks, recommendations, and similar-item lookup.
- **MLflow-Ready:** `mlflow` included in dependencies for experiment tracking integration.

## 🏗️ Architecture / How It Works

The codebase is organized into three layers: **models**, **training**, and **serving**.

```
ml-recommendation-engine-hybrid/
├── main.py                  # Root FastAPI app (placeholder inference)
├── Dockerfile               # Container build (python:3.9-slim + uvicorn)
├── requirements.txt         # torch, fastapi, sklearn, mlflow, etc.
├── src/
│   ├── models/
│   │   └── recommender.py   # Core model architectures
│   ├── training/
│   │   └── trainer.py       # Dataset + training loop
│   └── serving/
│       └── app.py           # Production-style FastAPI service
└── tests/
    └── test_models.py       # pytest suite for model forward passes
```

### Model Architectures (`src/models/recommender.py`)

| Model | Description |
|---|---|
| `MatrixFactorization` | Classic user/item embedding dot-product with per-user, per-item, and global bias terms. |
| `NeuralCollaborativeFiltering` | Concatenates user/item embeddings and passes them through an MLP (`128 → 64 → 32`) with BatchNorm, ReLU, and Dropout(0.2). |
| `TwoTowerModel` | Separate user and item towers producing L2-normalized embeddings for efficient approximate nearest-neighbor retrieval; score = cosine-style dot product. |
| `HybridRecommender` | The flagship model. Computes a **CF score** (Neural CF) and a **content score** (projected user embedding · item-content embedding), then fuses both through a small MLP (`2 → 16 → 1`) to produce the final rating prediction. |

### Training Pipeline (`src/training/trainer.py`)

- `InteractionDataset` wraps `(user_id, item_id, rating)` triples and optionally attaches per-item feature vectors for the hybrid model.
- `Trainer` handles the full loop: Adam optimizer, `MSELoss`, automatic CUDA/CPU selection, per-epoch training, and validation with **RMSE** and **MAE** metrics.
- `fit()` returns a history dict of training loss and validation metrics across epochs.

### Serving Layer (`src/serving/app.py`)

- `POST /recommend` — accepts `{user_id, num_recommendations}` and returns ranked item/score pairs.
- `GET /similar/{item_id}` — returns top-k similar items.
- `GET /health` — liveness probe.
- Models are loaded via `RecommendationService.load_model(path)` with `torch.load` on CPU.

## 🛠️ Prerequisites

Ensure you have the following installed in your environment before proceeding:
- **Python 3.9+**
- **pip** (or a virtual environment tool such as `venv`)
- **Docker** (optional, for containerized deployment)
- A CUDA-capable GPU is optional; the trainer auto-detects and falls back to CPU.

## 📦 Installation

Follow standard installation steps for `Python` to set up the project locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/Shivay00001/ml-recommendation-engine-hybrid.git
   ```
2. Navigate to the project directory:
   ```bash
   cd ml-recommendation-engine-hybrid
   ```
3. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## 🐳 Running with Docker

The repository ships with a `Dockerfile` based on `python:3.9-slim` that installs all dependencies and launches the API with **uvicorn**.

### Build the image

```bash
docker build -t ml-recommendation-engine-hybrid .
```

### Run the container

```bash
docker run -p 8000:8000 ml-recommendation-engine-hybrid
```

The API is then available at `http://localhost:8000` (interactive docs at `http://localhost:8000/docs`).

### Optional: docker-compose

If you prefer compose, create a `docker-compose.yml` like:

```yaml
version: "3.9"
services:
  api:
    build: .
    ports:
      - "8000:8000"
```

Then run:

```bash
docker-compose up --build
```

> **Note:** The `Dockerfile` runs `main:app` (the root placeholder app). To serve the full recommendation API instead, change the `CMD` to:
> ```dockerfile
> CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
> ```

## 💻 Usage

Run the project using standard execution commands for `Python`. Ensure all environment variables and configurations are set prior to execution.

### Start the API locally

```bash
uvicorn src.serving.app:app --reload --port 8000
```

### Request recommendations

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 42, "num_recommendations": 5}'
```

### Train a model (example)

```python
import numpy as np
from torch.utils.data import DataLoader
from src.models.recommender import HybridRecommender
from src.training.trainer import InteractionDataset, Trainer

users   = np.random.randint(0, 1000, size=10000)
items   = np.random.randint(0, 500, size=10000)
ratings = np.random.uniform(1, 5, size=10000)
feats   = np.random.randn(500, 32)   # per-item content features

dataset = InteractionDataset(users, items, ratings, feats)
loader  = DataLoader(dataset, batch_size=256, shuffle=True)

model   = HybridRecommender(num_users=1000, num_items=500, item_feature_dim=32)
trainer = Trainer(model, lr=0.001)
trainer.fit(loader, loader, epochs=10)
```

### Run tests

```bash
pytest tests/ -v
```

## 🔍 Workability Assessment

An honest evaluation of the repository's current state:

**What works well:**
- ✅ The model architectures are **correct and well-implemented** — all forward passes are covered by tests and produce expected tensor shapes.
- ✅ The `Trainer` is a solid, reusable training loop with proper device handling, evaluation metrics, and clean separation of concerns.
- ✅ The hybrid fusion design (CF score + content score → learned gating MLP) is a legitimate, industry-recognized pattern.
- ✅ Dependency pinning, `.gitignore` hygiene for secrets, and a working Dockerfile are in place.

**What is incomplete / requires fixing before production:**
- ⚠️ **Serving is stubbed.** `RecommendationService.get_recommendations()` returns random scores even when a model is loaded — real scoring over the item catalog is not implemented. `load_model` is never invoked at startup.
- ⚠️ **Two competing entrypoints.** `main.py` (used by the Dockerfile) is a placeholder that returns random vectors, while the real API lives in `src/serving/app.py`. The Docker CMD should be pointed at the latter.
- ⚠️ **No data pipeline.** There is no dataset ingestion, preprocessing, or feature-store code — you must supply interaction and item-feature arrays yourself.
- ⚠️ **No model persistence/checkpointing** in the trainer, and **MLflow is a dependency but is never used** in code.
- ⚠️ **No cold-start handling, negative sampling, or ranking metrics** (e.g., Recall@K, NDCG) — only RMSE/MAE regression metrics.
- ⚠️ The `TwoTowerModel` has no corresponding training or retrieval (ANN index) integration yet.

**Verdict:** This is a **strong foundation / reference implementation** — good model code and training infrastructure — but it is **not production-ready** as shipped. Expect to implement real inference logic, wire up the serving layer to trained checkpoints, and add a data pipeline before deploying it for real users.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 📝 License

This project is licensed under the **VisionQuantech Custom Commercial License** (see `LICENSE`):

- **Non-financial / personal / educational use:** Free.
- **Personal revenue-generating use:** Requires a 15–30% revenue share.
- **Business / enterprise use:** Requires a separate commercial license — contact **visionquantech@proton.me**.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.