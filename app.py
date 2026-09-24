"""FastAPI service: real hybrid recommendations from a trained numpy MF model."""

import os

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model import EXPECTED_COLS, MODEL_PATH, HybridRecommender

app = FastAPI(title="Hybrid Recommendation Engine", version="1.0.0")
_MODEL: HybridRecommender | None = None


def get_model() -> HybridRecommender:
    global _MODEL
    if _MODEL is None:
        if os.path.exists(MODEL_PATH):
            _MODEL = HybridRecommender.load(MODEL_PATH)
        else:
            raise HTTPException(
                status_code=503, detail="model not trained yet — POST /train first"
            )
    return _MODEL


def _load_items(items_csv_path: str | None, model: HybridRecommender) -> None:
    if items_csv_path and os.path.exists(items_csv_path):
        items = pd.read_csv(items_csv_path)
        if {"item_id", "title"}.issubset(items.columns):
            model.titles = dict(zip(items["item_id"], items["title"]))
        if "genre" in items.columns:
            model.genres = dict(zip(items["item_id"], items["genre"]))


class TrainRequest(BaseModel):
    ratings_csv_path: str
    items_csv_path: str | None = None


@app.get("/health")
def health():
    return {"status": "ok", "model_trained": os.path.exists(MODEL_PATH)}


@app.get("/schema")
def schema():
    return {"ratings_columns": EXPECTED_COLS,
            "items_columns": ["item_id", "title", "genre (optional)"]}


@app.post("/train")
def train(req: TrainRequest):
    if not os.path.exists(req.ratings_csv_path):
        raise HTTPException(status_code=404, detail="ratings CSV not found")
    global _MODEL
    ratings = pd.read_csv(req.ratings_csv_path)
    model = HybridRecommender()
    metrics = model.fit(ratings)
    _load_items(req.items_csv_path, model)
    model.save(MODEL_PATH)
    _MODEL = model
    return {"status": "trained", "model_path": MODEL_PATH, "metrics": metrics}


@app.get("/recommend")
def recommend(user_id: int, n: int = 10):
    try:
        return {"user_id": user_id, "recommendations": get_model().recommend(user_id, n)}
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/predict")
def predict(user_id: int, item_id: int):
    try:
        return get_model().predict(user_id, item_id)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8003)
