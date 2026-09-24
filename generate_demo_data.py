"""DEMO DATA GENERATOR — SYNTHETIC DATA, NOT REAL USER BEHAVIOR.

Small MovieLens-style dataset with a planted genre structure so recommendation
quality can be sanity-checked: users 1-5 love Action, users 6-10 love Comedy,
users 11-15 love Drama. NEVER present this as real ratings.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(3)

ITEMS = [
    (1, "Blade Protocol", "Action"), (2, "Iron Horizon", "Action"),
    (3, "Night Strike", "Action"), (4, "Zero Gravity Ops", "Action"),
    (13, "Crimson Tide Rising", "Action"), (14, "Final Stand", "Action"),
    (5, "Laugh Track", "Comedy"), (6, "The Prankster", "Comedy"),
    (7, "Office Chaos", "Comedy"), (8, "Stand-Up Nights", "Comedy"),
    (15, "Wedding Crashers 2", "Comedy"), (16, "Pajama Party", "Comedy"),
    (9, "The Long Goodbye", "Drama"), (10, "Silent Letters", "Drama"),
    (11, "Winter Promise", "Drama"), (12, "Half Light", "Drama"),
    (17, "The Last Letter", "Drama"), (18, "Ashes of Dawn", "Drama"),
]


def make_demo_csvs(ratings_path: str, items_path: str) -> tuple:
    genres = {i: g for i, _, g in ITEMS}
    rows = []
    for user in range(1, 16):
        taste = {1: "Action", 6: "Comedy", 11: "Drama"}[
            1 if user <= 5 else 6 if user <= 10 else 11
        ]
        for item_id, _, _ in ITEMS:
            if RNG.random() < 0.7:  # sparse like real data
                liked = genres[item_id] == taste
                base = 4.8 if liked else 1.6
                rating = float(np.clip(RNG.normal(base, 0.35), 0.5, 5.0))
                rows.append((user, item_id, round(rating, 1)))
    ratings = pd.DataFrame(rows, columns=["user_id", "item_id", "rating"])
    items = pd.DataFrame(ITEMS, columns=["item_id", "title", "genre"])
    ratings.to_csv(ratings_path, index=False)
    items.to_csv(items_path, index=False)
    return ratings_path, items_path


if __name__ == "__main__":
    make_demo_csvs("demo_ratings.csv", "demo_items.csv")
    print("wrote SYNTHETIC DEMO data -> demo_ratings.csv, demo_items.csv (NOT real ratings)")
