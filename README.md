# ML Recommendation Engine - Hybrid

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **production-grade hybrid recommendation engine** combining collaborative filtering, content-based filtering, and deep learning approaches.

## 🚀 Features

- **Collaborative Filtering**: Matrix factorization, Neural CF
- **Content-Based**: TF-IDF, embeddings
- **Deep Learning**: Two-tower architecture, attention
- **Real-time Serving**: FastAPI inference service
- **A/B Testing**: Experiment framework

## 📁 Structure

```
ml-recommendation-engine-hybrid/
├── src/
│   ├── models/           # ML models
│   ├── data/             # Data processing
│   ├── training/         # Training logic
│   ├── serving/          # API server
│   └── evaluation/       # Metrics
├── tests/
├── notebooks/
└── configs/
```

## 🛠️ Installation

```bash
pip install -r requirements.txt
python -m src.training.train --config configs/hybrid.yaml
python -m src.serving.app
```

## 📄 License

MIT License
