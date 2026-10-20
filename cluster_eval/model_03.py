"""
model_03.py  -  ONLY modelling + metrics. No plotting.

Metrics
  davies_bouldin_score    lower  = better (compact, well separated clusters)
  calinski_harabasz_score higher = better (between-cluster / within-cluster variance)
  silhouette_score        higher = better (extra reference metric)

DBSCAN labels noise as -1.  Noise points are removed before scoring and the
metrics are NaN if fewer than 2 clusters remain.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)


def evaluate(X, labels):
    mask = labels != -1
    n_clusters = len(set(labels[mask]))
    n_noise = int((~mask).sum())
    if n_clusters < 2:
        db = ch = sil = np.nan
    else:
        db = davies_bouldin_score(X[mask], labels[mask])
        ch = calinski_harabasz_score(X[mask], labels[mask])
        sil = silhouette_score(X[mask], labels[mask])
    return {"clusters": n_clusters, "noise": n_noise,
            "davies_bouldin": db, "calinski_harabasz": ch, "silhouette": sil}


def kmeans_k_sweep(X, k_range=range(2, 9)):
    """Evaluate KMeans for several k -> DataFrame (used to pick the best k)."""
    rows = []
    for k in k_range:
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
        rows.append({"k": k, **evaluate(X, labels)})
    return pd.DataFrame(rows)


def compare_algorithms(X, k, eps, min_samples=5):
    """KMeans vs Agglomerative vs DBSCAN on one dataset -> (DataFrame, labels dict)."""
    labels = {
        f"KMeans(k={k})": KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X),
        f"Agglomerative(k={k})": AgglomerativeClustering(n_clusters=k).fit_predict(X),
        f"DBSCAN(eps={eps})": DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X),
    }
    rows = [{"algorithm": name, **evaluate(X, lab)} for name, lab in labels.items()]
    return pd.DataFrame(rows), labels


if __name__ == "__main__":
    from load_01 import load_all
    from preprocess_02 import scale_all

    X, _ = scale_all(load_all())["blobs"]
    print(kmeans_k_sweep(X).round(3).to_string(index=False))
