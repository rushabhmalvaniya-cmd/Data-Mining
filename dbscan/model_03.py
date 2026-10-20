"""
model_03.py  -  ONLY modelling: k-distance, eps search, DBSCAN, KMeans. No plotting.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors

MIN_SAMPLES = 5


# ---------- k-distance graph data -------------------------------------------
def k_distances(X, k=MIN_SAMPLES):
    """Sorted distance of every point to its k-th nearest neighbour.

    DBSCAN counts the point itself in min_samples, so we ask for k neighbours
    (the first neighbour is the point itself at distance 0).
    """
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    dist, _ = nn.kneighbors(X)
    return np.sort(dist[:, -1])


def knee_eps(sorted_dist):
    """Elbow = point farthest from the straight line joining first and last point."""
    n = len(sorted_dist)
    # normalise both axes to [0, 1] so the x scale (n points) does not dominate
    x = np.linspace(0, 1, n)
    y = (sorted_dist - sorted_dist[0]) / (sorted_dist[-1] - sorted_dist[0])
    # distance below the straight line joining first and last point
    idx = int(np.argmax(x - y))
    return float(sorted_dist[idx]), idx


# ---------- clustering -------------------------------------------------------
def run_dbscan(X, eps=0.5, min_samples=MIN_SAMPLES):
    return DBSCAN(eps=eps, min_samples=min_samples).fit(X).labels_


def run_kmeans(X, k):
    return KMeans(n_clusters=k, n_init=10, random_state=42).fit(X).labels_


def summarize(labels):
    n_noise = int((labels == -1).sum())
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return n_clusters, n_noise


# ---------- eps tuning -------------------------------------------------------
def scan_eps(X, min_samples=MIN_SAMPLES, grid=None):
    """Try many eps values; return DataFrame(eps, n_clusters, n_noise)."""
    if grid is None:
        grid = np.round(np.arange(0.05, 1.01, 0.01), 2)
    rows = []
    for eps in grid:
        n_clusters, n_noise = summarize(run_dbscan(X, eps, min_samples))
        rows.append((float(eps), n_clusters, n_noise))
    return pd.DataFrame(rows, columns=["eps", "n_clusters", "n_noise"])


def best_eps_for_k(scan_df, k=3):
    """Among eps values giving exactly k clusters, keep those with the fewest
    noise points and return the SMALLEST one (the tightest eps that still works,
    so nearby clusters are not merged) + the range of eps that work."""
    ok = scan_df[scan_df.n_clusters == k]
    if ok.empty:
        return None
    best = ok[ok.n_noise == ok.n_noise.min()]
    chosen = float(best.eps.min())
    return {
        "eps": chosen,
        "n_noise": int(best.n_noise.min()),
        "stable_range": (float(best.eps.min()), float(best.eps.max())),
    }


if __name__ == "__main__":
    from load_01 import load_all
    from preprocess_02 import scale_all

    data = scale_all(load_all())
    X, _ = data["blobs"]
    sd = k_distances(X)
    eps, _ = knee_eps(sd)
    print("blobs knee eps:", round(eps, 3))
    print("blobs best eps for 3 clusters:", best_eps_for_k(scan_eps(X), 3))
