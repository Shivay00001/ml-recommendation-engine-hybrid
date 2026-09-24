# ml-recommendation-engine-hybrid

**Real** hybrid recommendation engine (rebuilt 2026-09-24 — the previous version
was a simulated-inference stub and has been deleted). Trains genuine matrix
factorization with SGD in **pure numpy** (no torch):

`r̂(u,i) = μ + b_u + b_i + p_u · q_i`, blended with an item-popularity signal
(mean rating per item). Every recommendation is computed from parameters learned
on real input ratings — nothing canned, nothing random at serving time.

## Input format (MovieLens-style)

`ratings.csv`: `user_id,item_id,rating`
`items.csv` (optional): `item_id,title,genre`

## Quickstart

```bash
pip install -r requirements.txt
python generate_demo_data.py   # SYNTHETIC demo ratings — for testing only
uvicorn app:app --port 8003
curl -X POST localhost:8003/train -H 'Content-Type: application/json' \
  -d '{"ratings_csv_path": "demo_ratings.csv", "items_csv_path": "demo_items.csv"}'
curl 'localhost:8003/recommend?user_id=1&n=5'
curl 'localhost:8003/predict?user_id=1&item_id=2'
pytest -q
```

## Honest limits

- Demo ratings are **synthetic** with a planted genre structure; real quality
  depends on real rating data and sparsity.
- Cold-start users/items (never seen in training) return an error rather than a
  fake guess.
- Train RMSE is in-sample; hold out ratings for a real evaluation.
