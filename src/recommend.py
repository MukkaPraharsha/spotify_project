"""Song recommender CLI.  Usage:
   python src/recommend.py "Shape of You" --artist "Ed Sheeran" -n 10 --same-genre
"""
import os, argparse, joblib, numpy as np, pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
b = joblib.load(f"{ROOT}/models/recommender.joblib"); m, X = b["meta"], b["X"]
ap = argparse.ArgumentParser(); ap.add_argument("song"); ap.add_argument("--artist"); ap.add_argument("-n", type=int, default=10)
ap.add_argument("--same-genre", action="store_true"); ap.add_argument("--any-cluster", action="store_true"); a = ap.parse_args()
mask = m.track_name.str.lower().str.contains(a.song.lower(), regex=False)
if a.artist: mask &= m.track_artist.str.lower().str.contains(a.artist.lower(), regex=False)
if not mask.any(): raise SystemExit("Song not found")
i = m[mask].index[0]; cand = np.arange(len(m)) if a.any_cluster else np.where(m.cluster == m.cluster[i])[0]
if a.same_genre: cand = cand[m.playlist_genre.values[cand] == m.playlist_genre[i]]
cand = cand[cand != i]
sims = cosine_similarity(X[i:i + 1], X[cand]).ravel()
out = m.loc[cand].assign(similarity=sims).sort_values("similarity", ascending=False).head(a.n)
print(f"Query: {m.track_name[i]} - {m.track_artist[i]} [{m.playlist_genre[i]} | {m.cluster_name[i]}]\n")
print(out[["track_name", "track_artist", "playlist_genre", "similarity"]].round(3).to_string(index=False))
