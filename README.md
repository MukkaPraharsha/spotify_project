# Spotify Songs' Genre Segmentation & Recommendation System

Groups ~26,000 Spotify tracks into clusters by their audio features, compares the clusters with the playlist genres,
subgenres and playlist names, and uses the result to power a content-based song recommender.

## Dataset
`data/spotify_dataset.csv` - 32,833 rows x 23 columns: track/artist/album info, `playlist_name`, `playlist_genre`
(6: edm, rap, pop, r&b, latin, rock), `playlist_subgenre` (24), `track_popularity`, and audio features
(danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration).

## Structure
```
spotify_project/
├── data/      spotify_dataset.csv (raw), spotify_clean.csv, songs_with_clusters.csv
├── figures/   21 plots
├── models/    recommender.joblib (scaler, K-Means, features, vectors), playlist_clusters.csv, summary.json
├── src/       spotify_analysis.py (full pipeline), recommend.py (CLI recommender)
├── Spotify_Genre_Segmentation.ipynb   (same pipeline; run from project root)
├── requirements.txt, run_log.txt
```
Run: `pip install -r requirements.txt && python src/spotify_analysis.py` (~3 min), then
`python src/recommend.py "Blinding Lights" --artist "The Weeknd" -n 10 [--same-genre] [--any-cluster]`.

## What was done (maps to the brief)

**1. Pre-processing** - removed 5 rows with missing track info; the same track appears in several playlists, so duplicates by
`track_id` and by (name, artist) were dropped: **32,833 -> 26,229 unique songs**. Parsed release year, converted duration to minutes,
log-transformed skewed features (speechiness, acousticness, instrumentalness, liveness), standardised everything before clustering.

**2. EDA (figs 01-09)** - genre/subgenre/artist/playlist counts, feature distributions, features by genre, genre "fingerprints"
(heatmap + radar), scatter pairs, popularity & release-year trends, key/mode, pairplot.
Insights (genre means): rap has by far the highest speechiness (0.20 vs <=0.12 elsewhere); EDM is the most energetic (0.81) and loudest;
latin and rap are the most danceable (0.71) and latin is the happiest (valence 0.61); rock is the least danceable (0.52);
r&b is the lowest-energy and quietest genre (0.58, -8.0 dB); EDM has the lowest valence (0.40). Pop sits close to the overall average.
See `04_genre_profile_heatmap.png` for all features.

**3. Correlation matrix (fig 10)** - strongest: loudness-energy **+0.69**, acousticness-energy **-0.55**, acousticness-loudness **-0.38**,
valence-danceability **+0.34**. Popularity is only weakly related to any single audio feature.

**4. Clustering (figs 11-20)** - K-Means on 10 standardised audio features. Elbow/silhouette/Davies-Bouldin were checked for k=2-12
(fig 11); the metrics have no sharp optimum, so **k=6** was chosen to match the six genres and give interpretable groups.

| Cluster | Songs | Character | Dominant genres |
|---|---|---|---|
| C0 | 6,424 | happy, danceable, non-speechy | pop, latin, r&b |
| C1 | 5,391 | high energy, low danceability, sad/angry (low valence) | edm, rock, pop |
| C2 | 3,203 | instrumental, high energy, long tracks | edm |
| C3 | 4,076 | quiet, acoustic, low energy | r&b, rock, pop |
| C4 | 2,722 | very fast tempo (~166 BPM), speechy | rap |
| C5 | 4,413 | speechy and highly danceable | rap, r&b |

Clusters were plotted against **playlist genre** (figs 13-16), **subgenre** (17) and **playlist names** (18-19) using PCA and t-SNE.
Additionally, 402 playlists with at least 20 songs were clustered by their average audio profile (fig 20, `models/playlist_clusters.csv`):
e.g. one group is mostly 90s hip-hop/gangsta-rap playlists and another is mostly classic-rock/70s playlists (printed in `run_log.txt`).

**Honest assessment of cluster quality:** silhouette is only ~0.12 and agreement with genre labels is modest
(NMI 0.12, ARI 0.08). Genres overlap heavily in audio space (pop/latin/r&b especially), so clusters describe *sound*
(e.g. "quiet acoustic", "speechy + danceable") rather than reproducing playlist genres. Agglomerative clustering did no better (NMI 0.10).

**5. Recommendation system** - content-based: cosine similarity between standardised audio vectors, searching inside the
query song's cluster (optionally also restricted to the same genre). Example: `Shape of You` with `--same-genre` returns pop tracks with
similar profiles.

Evaluation (genre precision@10 over 1,500 random query songs): random = 17%, audio-similarity = **38%**
(cluster-restricted 38%). So audio similarity is about 2.2x better than chance at finding same-genre songs, but audio alone does not
identify genre. Cluster restriction does not hurt accuracy and makes lookups faster. The `--same-genre` option is 100% same-genre by construction, so that
figure is not an accuracy measure.

## Limitations & next steps
- Audio features only: no lyrics, artist, language, or listening history, so recs can cross genres/languages (e.g. latin vs pop).
- `playlist_genre` is the playlist's label, not necessarily the song's true genre.
- Improvements: add artist/genre metadata as a hybrid score, a popularity boost (`popularity_weight` is supported in the code), collaborative filtering if user data is available, UMAP/GMM clustering.
