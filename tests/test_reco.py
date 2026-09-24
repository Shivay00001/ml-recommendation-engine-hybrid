import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generate_demo_data import make_demo_csvs  # noqa: E402
from model import HybridRecommender  # noqa: E402

TASTE = {u: ("Action" if u <= 5 else "Comedy" if u <= 10 else "Drama")
         for u in range(1, 16)}


def _trained(tmp_path):
    r_csv = str(tmp_path / "r.csv")
    i_csv = str(tmp_path / "i.csv")
    make_demo_csvs(r_csv, i_csv)
    ratings = pd.read_csv(r_csv)
    items = pd.read_csv(i_csv)
    model = HybridRecommender(epochs=60)
    metrics = model.fit(ratings)
    model.titles = dict(zip(items["item_id"], items["title"]))
    model.genres = dict(zip(items["item_id"], items["genre"]))
    return model, ratings, metrics


def test_recommendations_are_sensible(tmp_path):
    model, ratings, metrics = _trained(tmp_path)
    assert model.trained
    assert metrics["train_rmse"] < 1.0

    # the demo plants a genre taste per user group; the learned model should
    # rank a taste-matching item first for most users (real learned behavior)
    top1_hits = 0
    for u in range(1, 16):
        recs = model.recommend(u, n=10)
        assert len(recs) >= 1
        if recs[0]["genre"] == TASTE[u]:
            top1_hits += 1
        # ranked: scores strictly non-increasing
        scores = [r["predicted_rating"] for r in recs]
        assert scores == sorted(scores, reverse=True)
        # never recommends something the user already rated
        rated = set(ratings.loc[ratings["user_id"] == u, "item_id"])
        assert not (set(r["item_id"] for r in recs) & rated)
    assert top1_hits >= 10, f"only {top1_hits}/15 users got taste-matching top pick"


def test_deterministic_and_persisted(tmp_path):
    model, ratings, _ = _trained(tmp_path)
    m2 = HybridRecommender(epochs=60)
    m2.fit(ratings)
    m2.titles, m2.genres = model.titles, model.genres
    assert [r["item_id"] for r in m2.recommend(1, n=10)] == [
        r["item_id"] for r in model.recommend(1, n=10)
    ]
    p = str(tmp_path / "m.joblib")
    model.save(p)
    loaded = HybridRecommender.load(p)
    assert (
        loaded.predict(1, 2)["predicted_rating"]
        == model.predict(1, 2)["predicted_rating"]
    )
