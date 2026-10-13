# %% [markdown]
# # Spotify Songs' Genre Segmentation & Recommendation System
# Preprocessing -> EDA -> correlation -> K-Means clustering (vs genres / playlists) -> recommender.

# %%
import os, json, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="Set2")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG, MOD = f"{ROOT}/figures", f"{ROOT}/models"
RS = 42
def save(n): plt.tight_layout(); plt.savefig(f"{FIG}/{n}.png", dpi=110); plt.close()

# %% [markdown]
# ## 1. Load & pre-process
# %%
df = pd.read_csv(f"{ROOT}/data/spotify_dataset.csv")
print(df.shape); print(df.isnull().sum()[lambda s: s > 0])
raw_n = len(df)
df = df.dropna(subset=["track_name", "track_artist"])                       # rows with no track info
df = df.drop_duplicates(subset="track_id")                                  # same track listed in several playlists
df = df.drop_duplicates(subset=["track_name", "track_artist"]).reset_index(drop=True)   # re-releases of same song
df["release_year"] = pd.to_datetime(df.track_album_release_date, errors="coerce", format="mixed").dt.year
df["duration_min"] = df.duration_ms / 60000
print(f"rows: raw={raw_n} -> clean={len(df)}")
df.to_csv(f"{ROOT}/data/spotify_clean.csv", index=False)
audio = ["danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo", "duration_min"]
print(df[audio + ["track_popularity"]].describe().T.round(3))

# %% [markdown]
# ## 2. Exploratory data analysis
# %%
fig, axs = plt.subplots(2, 3, figsize=(16, 9))
sns.countplot(y="playlist_genre", data=df, order=df.playlist_genre.value_counts().index, ax=axs[0, 0]); axs[0, 0].set_title("Songs per genre")
sub = df.playlist_subgenre.value_counts()
sns.barplot(x=sub.values, y=sub.index, ax=axs[0, 1]); axs[0, 1].set_title("Songs per subgenre"); axs[0, 1].tick_params(labelsize=7)
sns.histplot(df.track_popularity, bins=40, ax=axs[0, 2]); axs[0, 2].set_title("Popularity distribution")
df.release_year.dropna().astype(int).value_counts().sort_index().loc[1990:].plot(ax=axs[1, 0]); axs[1, 0].set_title("Songs by release year")
top = df.track_artist.value_counts().head(15)
sns.barplot(x=top.values, y=top.index, ax=axs[1, 1]); axs[1, 1].set_title("Top 15 artists by tracks")
pn = df.playlist_name.value_counts().head(15)
sns.barplot(x=pn.values, y=pn.index, ax=axs[1, 2]); axs[1, 2].set_title("Top 15 playlists"); axs[1, 2].tick_params(labelsize=7)
save("01_overview")

fig, axs = plt.subplots(3, 4, figsize=(18, 11))
for a, c in zip(axs.ravel(), audio + ["track_popularity", "key"]):
    sns.histplot(df[c], kde=True, ax=a); a.set_title(c)
save("02_feature_distributions")

fig, axs = plt.subplots(2, 5, figsize=(22, 9))
for a, c in zip(axs.ravel(), audio):
    sns.boxplot(x="playlist_genre", y=c, data=df, ax=a); a.set_title(f"{c} by genre"); a.set_xlabel(""); a.tick_params(axis="x", rotation=40)
save("03_features_by_genre")

gm = df.groupby("playlist_genre")[audio[:8]].mean()
plt.figure(figsize=(10, 5)); sns.heatmap(((gm - gm.mean()) / gm.std()).T, annot=gm.T.round(2), fmt="", cmap="RdBu_r", center=0, cbar_kws={"label": "z-score across genres"})
plt.title("Average audio profile per genre"); save("04_genre_profile_heatmap")

# radar per genre
mm = (gm - gm.min()) / (gm.max() - gm.min())
ang = np.linspace(0, 2 * np.pi, len(mm.columns), endpoint=False).tolist(); ang += ang[:1]
plt.figure(figsize=(8, 8)); ax = plt.subplot(111, polar=True)
for g, r in mm.iterrows():
    v = r.tolist() + r.tolist()[:1]; ax.plot(ang, v, label=g); ax.fill(ang, v, alpha=.08)
ax.set_xticks(ang[:-1]); ax.set_xticklabels(mm.columns, fontsize=8); ax.set_title("Genre audio fingerprints (min-max scaled)"); ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
save("05_genre_radar")

fig, axs = plt.subplots(1, 3, figsize=(18, 5))
sns.scatterplot(x="energy", y="valence", hue="playlist_genre", data=df.sample(4000, random_state=RS), alpha=.5, ax=axs[0]); axs[0].set_title("Energy vs valence")
sns.scatterplot(x="energy", y="loudness", hue="playlist_genre", data=df.sample(4000, random_state=RS), alpha=.5, ax=axs[1]); axs[1].set_title("Energy vs loudness")
sns.scatterplot(x="acousticness", y="energy", hue="playlist_genre", data=df.sample(4000, random_state=RS), alpha=.5, ax=axs[2]); axs[2].set_title("Acousticness vs energy")
save("06_scatter_pairs")

fig, axs = plt.subplots(1, 3, figsize=(18, 5))
sns.boxplot(x="playlist_genre", y="track_popularity", data=df, ax=axs[0]); axs[0].set_title("Popularity by genre")
yr = df.dropna(subset=["release_year"]); yr = yr[yr.release_year >= 1980]
yr.groupby("release_year")[["energy", "danceability", "valence", "acousticness"]].mean().plot(ax=axs[1]); axs[1].set_title("Mean audio features by release year")
sns.countplot(x="key", hue="mode", data=df, ax=axs[2]); axs[2].set_title("Key & mode (0=minor, 1=major)")
save("07_popularity_trends_keys")

sg = df.groupby(["playlist_genre", "playlist_subgenre"]).size().reset_index(name="n")
fig = plt.figure(figsize=(10, 6)); sns.barplot(x="n", y="playlist_subgenre", hue="playlist_genre", data=sg, dodge=False); plt.title("Subgenres grouped by genre"); save("08_subgenres")

g = sns.pairplot(df.sample(2000, random_state=RS)[["danceability", "energy", "valence", "acousticness", "playlist_genre"]], hue="playlist_genre", corner=True, plot_kws={"alpha": .4, "s": 10})
g.figure.savefig(f"{FIG}/09_pairplot.png", dpi=80); plt.close("all")

# %% [markdown]
# ## 3. Correlation matrix
# %%
corr = df[audio + ["track_popularity", "key", "mode"]].corr()
plt.figure(figsize=(12, 10)); sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, mask=np.triu(np.ones_like(corr, dtype=bool), 1), square=True)
plt.title("Correlation matrix of audio features"); save("10_correlation_matrix")
cp = corr.where(np.tril(np.ones(corr.shape, dtype=bool), -1)).stack().dropna().sort_values()
print("Strongest negative:\n", cp.head(4).round(3), "\nStrongest positive:\n", cp.tail(4).round(3))

# %% [markdown]
# ## 4. Clustering
# Features are standardised; skewed ones (speechiness, instrumentalness, liveness, acousticness) are log-transformed.
# %%
cl = df[audio].copy()
for c in ["speechiness", "instrumentalness", "liveness", "acousticness"]: cl[c] = np.log1p(cl[c] * 100)
scaler = StandardScaler(); Xs = scaler.fit_transform(cl)

ks = range(2, 13); inertia, sil, dbi = [], [], []
for k in ks:
    km = KMeans(k, n_init=10, random_state=RS).fit(Xs); inertia.append(km.inertia_)
    sil.append(silhouette_score(Xs, km.labels_, sample_size=8000, random_state=RS)); dbi.append(davies_bouldin_score(Xs, km.labels_))
fig, axs = plt.subplots(1, 3, figsize=(17, 4.5))
axs[0].plot(ks, inertia, "o-"); axs[0].set_title("Elbow (inertia)")
axs[1].plot(ks, sil, "o-", c="g"); axs[1].set_title("Silhouette (higher better)")
axs[2].plot(ks, dbi, "o-", c="r"); axs[2].set_title("Davies-Bouldin (lower better)")
for a in axs: a.set_xlabel("k")
save("11_choose_k")
print(pd.DataFrame({"k": list(ks), "inertia": inertia, "silhouette": sil, "DB": dbi}).round(3).to_string(index=False))

K = 6   # matches the 6 playlist genres; also compared against the silhouette-optimal k below
km = KMeans(K, n_init=20, random_state=RS).fit(Xs)
df["cluster"] = km.labels_
print("K-Means k=6: silhouette", round(silhouette_score(Xs, km.labels_, sample_size=8000, random_state=RS), 3),
      "ARI vs genre", round(adjusted_rand_score(df.playlist_genre, df.cluster), 3),
      "NMI vs genre", round(normalized_mutual_info_score(df.playlist_genre, df.cluster), 3))
sub_s = np.random.RandomState(RS).choice(len(df), 6000, replace=False)
agg = AgglomerativeClustering(K).fit_predict(Xs[sub_s])
print("Agglomerative (6k sample) NMI vs genre:", round(normalized_mutual_info_score(df.playlist_genre.iloc[sub_s], agg), 3),
      "| K-Means same sample:", round(normalized_mutual_info_score(df.playlist_genre.iloc[sub_s], km.labels_[sub_s]), 3))

# cluster profiles + auto-naming
prof = df.groupby("cluster")[audio].mean()
z = (prof - df[audio].mean()) / df[audio].std()
def name_cluster(r):
    t = r.abs().sort_values(ascending=False).index[:2]
    return " & ".join(("high " if r[c] > 0 else "low ") + c for c in t)
names = {c: f"C{c}: {name_cluster(z.loc[c])}" for c in z.index}
df["cluster_name"] = df.cluster.map(names)
print(pd.concat([prof.round(2), df.cluster.value_counts().sort_index().rename("n")], axis=1).to_string()); print(names)
plt.figure(figsize=(11, 5)); sns.heatmap(z.T, annot=prof.T.round(2), fmt="", cmap="RdBu_r", center=0, xticklabels=[names[c] for c in z.index]); plt.xticks(rotation=25, ha="right", fontsize=7)
plt.title("Cluster audio profiles (colour = z-score)"); save("12_cluster_profiles")

# PCA & t-SNE
pca = PCA(2, random_state=RS); P = pca.fit_transform(Xs)
df["pc1"], df["pc2"] = P[:, 0], P[:, 1]
print("PCA explained variance:", pca.explained_variance_ratio_.round(3))
fig, axs = plt.subplots(1, 2, figsize=(17, 7)); s = df.sample(8000, random_state=RS)
sns.scatterplot(x="pc1", y="pc2", hue="cluster", palette="tab10", data=s, s=12, alpha=.6, ax=axs[0]); axs[0].set_title("PCA - K-Means clusters")
sns.scatterplot(x="pc1", y="pc2", hue="playlist_genre", data=s, s=12, alpha=.6, ax=axs[1]); axs[1].set_title("PCA - playlist genre")
save("13_pca_clusters_vs_genre")
ts = np.random.RandomState(RS).choice(len(df), 5000, replace=False)
T = TSNE(2, perplexity=40, random_state=RS, init="pca").fit_transform(Xs[ts])
fig, axs = plt.subplots(1, 2, figsize=(17, 7))
sns.scatterplot(x=T[:, 0], y=T[:, 1], hue=df.cluster.iloc[ts].values, palette="tab10", s=10, ax=axs[0]); axs[0].set_title("t-SNE - clusters")
sns.scatterplot(x=T[:, 0], y=T[:, 1], hue=df.playlist_genre.iloc[ts].values, s=10, ax=axs[1]); axs[1].set_title("t-SNE - playlist genre")
save("14_tsne_clusters_vs_genre")

# clusters vs genre / subgenre / playlist names
ct = pd.crosstab(df.playlist_genre, df.cluster_name, normalize="index") * 100
plt.figure(figsize=(11, 5)); sns.heatmap(ct, annot=True, fmt=".0f", cmap="YlGnBu"); plt.xticks(rotation=25, ha="right", fontsize=7); plt.title("% of each genre falling in each cluster"); save("15_clusters_vs_genre")
ct2 = pd.crosstab(df.cluster_name, df.playlist_genre, normalize="index") * 100
ct2.plot(kind="barh", stacked=True, figsize=(11, 5)); plt.title("Genre make-up of each cluster (%)"); save("16_cluster_genre_composition")
cs = pd.crosstab(df.playlist_subgenre, df.cluster, normalize="index") * 100
plt.figure(figsize=(9, 9)); sns.heatmap(cs, annot=True, fmt=".0f", cmap="YlOrRd"); plt.title("% of each subgenre in each cluster"); save("17_clusters_vs_subgenre")
top_pl = df.playlist_name.value_counts().head(25).index
cp_ = pd.crosstab(df[df.playlist_name.isin(top_pl)].playlist_name, df.cluster, normalize="index") * 100
plt.figure(figsize=(9, 10)); sns.heatmap(cp_, annot=True, fmt=".0f", cmap="PuBu"); plt.title("% of each of the 25 biggest playlists in each cluster"); save("18_clusters_vs_playlist_names")
pm = df[df.playlist_name.isin(df.playlist_name.value_counts().head(40).index)].groupby("playlist_name")[["energy", "danceability", "valence", "acousticness"]].mean()
sns.clustermap(pm, standard_scale=1, cmap="viridis", figsize=(8, 12)).figure.savefig(f"{FIG}/19_playlist_name_clustermap.png", dpi=100); plt.close("all")

# playlist-level clustering: group the 449 playlists by their mean audio profile
pl = df.groupby("playlist_name")[audio].mean()
pl = pl[df.playlist_name.value_counts().reindex(pl.index) >= 20]
plz = StandardScaler().fit_transform(pl); pk = KMeans(5, n_init=10, random_state=RS).fit(plz)
pl["playlist_cluster"] = pk.labels_; pp = PCA(2, random_state=RS).fit_transform(plz)
plt.figure(figsize=(10, 7)); sns.scatterplot(x=pp[:, 0], y=pp[:, 1], hue=pk.labels_, palette="tab10", s=60)
for i in np.random.RandomState(RS).choice(len(pl), 25, replace=False): plt.annotate(pl.index[i][:22], (pp[i, 0], pp[i, 1]), fontsize=7)
plt.title(f"{len(pl)} playlists clustered by average audio features"); save("20_playlist_clusters")
pl.round(3).to_csv(f"{MOD}/playlist_clusters.csv")
for c in range(5): print("playlist cluster", c, "->", list(pl[pl.playlist_cluster == c].index[:6]))

# %% [markdown]
# ## 5. Recommendation system
# Content-based: cosine similarity on the scaled audio vector, restricted to the same cluster first (fast + coherent),
# falling back to the full catalogue. Evaluation: how often recommended songs share the query's genre (precision@10).
# %%
meta = df[["track_id", "track_name", "track_artist", "playlist_genre", "playlist_subgenre", "playlist_name", "track_popularity", "cluster", "cluster_name"]].reset_index(drop=True)
class Recommender:
    def __init__(self, meta, X): self.m, self.X = meta, X
    def recommend(self, song, artist=None, n=10, same_cluster=True, same_genre=False, popularity_weight=0.0):
        mask = self.m.track_name.str.lower().str.contains(song.lower(), regex=False)
        if artist: mask &= self.m.track_artist.str.lower().str.contains(artist.lower(), regex=False)
        if not mask.any(): raise ValueError(f"'{song}' not found")
        i = self.m[mask].index[0]; cand = self.m.index[self.m.cluster == self.m.cluster[i]] if same_cluster else self.m.index
        if same_genre: cand = cand[self.m.playlist_genre.values[cand] == self.m.playlist_genre[i]]
        sims = cosine_similarity(self.X[i:i + 1], self.X[cand]).ravel()
        if popularity_weight: sims = sims + popularity_weight * self.m.track_popularity.values[cand] / 100
        out = self.m.loc[cand].assign(similarity=sims).drop(index=i)
        out = out[~((out.track_name == self.m.track_name[i]) & (out.track_artist == self.m.track_artist[i]))]
        return self.m.loc[[i]], out.sort_values("similarity", ascending=False).head(n)
rec = Recommender(meta, Xs)
q, r = rec.recommend("Shape of You", n=10, same_genre=True)
print("QUERY:", q[["track_name", "track_artist", "playlist_genre"]].to_string(index=False)); print(r[["track_name", "track_artist", "playlist_genre", "similarity"]].round(3).to_string(index=False))

rng = np.random.RandomState(RS); qs = rng.choice(len(meta), 1500, replace=False); res = {"audio only (full catalogue)": [], "cluster-restricted": [], "cluster + same genre": [], "random": []}
for i in qs:
    g = meta.playlist_genre[i]
    for key, mask in [("cluster-restricted", meta.cluster == meta.cluster[i]), ("audio only (full catalogue)", np.ones(len(meta), bool)),
                      ("cluster + same genre", (meta.cluster == meta.cluster[i]) & (meta.playlist_genre == g))]:
        idx = np.where(mask)[0]; idx = idx[idx != i]; s = cosine_similarity(Xs[i:i + 1], Xs[idx]).ravel()
        res[key].append((meta.playlist_genre.values[idx[np.argsort(-s)[:10]]] == g).mean())
    res["random"].append((meta.playlist_genre.values[rng.choice(len(meta), 10)] == g).mean())
ev = {k: round(float(np.mean(v)), 4) for k, v in res.items()}; print("Genre precision@10:", ev)
plt.figure(figsize=(8, 4)); ev_plot = {k: v for k, v in ev.items() if k != "cluster + same genre"}  # that one is 1.0 by construction (it filters on genre)
sns.barplot(x=list(ev_plot.keys()), y=list(ev_plot.values())); plt.xticks(rotation=15); plt.ylabel("genre precision@10"); plt.title("Recommender evaluation"); save("21_recommender_eval")

joblib.dump({"scaler": scaler, "kmeans": km, "audio_features": audio, "cluster_names": names, "meta": meta, "X": Xs}, f"{MOD}/recommender.joblib")
df[["track_id", "track_name", "track_artist", "playlist_name", "playlist_genre", "playlist_subgenre", "cluster", "cluster_name"]].to_csv(f"{ROOT}/data/songs_with_clusters.csv", index=False)
json.dump({"n_songs": len(df), "k": K, "cluster_names": names, "genre_precision_at_10": ev}, open(f"{MOD}/summary.json", "w"), indent=2)
print("saved")
