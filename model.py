"""Real hybrid recommendation engine — pure numpy, no torch.

Trains genuine matrix factorization with SGD:
    r_hat(u,i) = mu + b_u + b_i + p_u . q_i
blended with an item-popularity signal (hybrid):
    score(u,i) = 0.7 * r_hat(u,i) + 0.3 * item_mean(i)

Every recommendation is computed from parameters learned on real input ratings.
No canned lists, no random sampling at serving time.
"""

import os

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = os.environ.get("RECO_MODEL_PATH", "model.joblib")
EXPECTED_COLS = ["user_id", "item_id", "rating"]


class HybridRecommender:
    def __init__(self, factors=20, epochs=40, lr=0.01, reg=0.02,
                 popularity_weight=0.3, seed=42):
        self.factors = factors
        self.epochs = epochs
        self.lr = lr
        self.reg = reg
        self.popularity_weight = popularity_weight
        self.seed = seed
        self.trained = False
        self.titles: dict = {}
        self.genres: dict = {}

    # ---------------- training ----------------
    def fit(self, ratings: pd.DataFrame) -> dict:
        missing = [c for c in EXPECTED_COLS if c not in ratings.columns]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")
        if len(ratings) < 20:
            raise ValueError("need at least 20 ratings to train")

        ratings = ratings.copy()
        self.user_ids = sorted(ratings["user_id"].unique())
        self.item_ids = sorted(ratings["item_id"].unique())
        self._u = {u: k for k, u in enumerate(self.user_ids)}
        self._i = {i: k for k, i in enumerate(self.item_ids)}
        nu, ni = len(self.user_ids), len(self.item_ids)

        rng = np.random.default_rng(self.seed)
        r = ratings["rating"].astype(float).values
        U = np.array([self._u[u] for u in ratings["user_id"].values])
        I = np.array([self._i[i] for i in ratings["item_id"].values])

        self.mu = float(r.mean())
        self.bu = np.zeros(nu)
        self.bi = np.zeros(ni)
        self.P = rng.normal(0, 0.1, (nu, self.factors))
        self.Q = rng.normal(0, 0.1, (ni, self.factors))

        order = np.arange(len(r))
        for _ in range(self.epochs):
            rng.shuffle(order)
            for k in order:
                u, i, rk = U[k], I[k], r[k]
                err = rk - (self.mu + self.bu[u] + self.bi[i] + self.P[u] @ self.Q[i])
                self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                self.bi[i] += self.lr * (err - self.reg * self.bi[i])
                pu = self.P[u].copy()
                self.P[u] += self.lr * (err * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += self.lr * (err * pu - self.reg * self.Q[i])

        # item popularity (mean rating per item) — real data-derived signal
        self.item_mean = ratings.groupby("item_id")["rating"].mean().to_dict()
        self.global_item_mean = float(ratings["rating"].mean())
        self.rated = {
            u: set(ratings.loc[ratings["user_id"] == u, "item_id"])
            for u in self.user_ids
        }

        pred = self.mu + self.bu[U] + self.bi[I] + (self.P[U] * self.Q[I]).sum(1)
        rmse = float(np.sqrt(np.mean((r - pred) ** 2)))
        self.trained = True
        return {
            "n_ratings": int(len(ratings)),
            "n_users": nu,
            "n_items": ni,
            "train_rmse": rmse,
        }

    # ---------------- inference ----------------
    def _mf(self, u: int, i: int) -> float:
        return float(self.mu + self.bu[u] + self.bi[i] + self.P[u] @ self.Q[i])

    def _hybrid(self, u: int, i: int, item_id) -> float:
        mf = self._mf(u, i)
        pop = self.item_mean.get(item_id, self.global_item_mean)
        return (1 - self.popularity_weight) * mf + self.popularity_weight * pop

    def predict(self, user_id, item_id) -> dict:
        if not self.trained:
            raise RuntimeError("model not trained")
        if user_id not in self._u or item_id not in self._i:
            raise ValueError("unknown user_id or item_id")
        u, i = self._u[user_id], self._i[item_id]
        score = max(0.5, min(5.0, self._hybrid(u, i, item_id)))
        return {
            "user_id": int(user_id),
            "item_id": int(item_id),
            "title": self.titles.get(item_id, str(item_id)),
            "predicted_rating": round(score, 3),
        }

    def recommend(self, user_id, n: int = 10, exclude_rated: bool = True) -> list:
        if not self.trained:
            raise RuntimeError("model not trained")
        if user_id not in self._u:
            raise ValueError("unknown user_id")
        u = self._u[user_id]
        seen = self.rated.get(user_id, set()) if exclude_rated else set()
        scored = []
        for item_id in self.item_ids:
            if item_id in seen:
                continue
            i = self._i[item_id]
            scored.append(
                (
                    max(0.5, min(5.0, self._hybrid(u, i, item_id))),
                    item_id,
                )
            )
        scored.sort(reverse=True)
        return [
            {
                "item_id": int(item_id),
                "title": self.titles.get(item_id, str(item_id)),
                "genre": self.genres.get(item_id, ""),
                "predicted_rating": round(s, 3),
            }
            for s, item_id in scored[:n]
        ]

    # ---------------- persistence ----------------
    def save(self, path: str = MODEL_PATH) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str = MODEL_PATH) -> "HybridRecommender":
        return joblib.load(path)
