"""Clustering basics: K-Means (from scratch + sklearn), Elbow, Silhouette, Hierarchical.
Run: python clustering_basics.py      Needs: numpy, scikit-learn, scipy, matplotlib
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# ---- 1. Data: six points, two features ----
X = np.array([[1, 1], [1.5, 2], [3, 4], [5, 7], [3.5, 5], [4.5, 5]])

# ---- 2. K-Means from scratch (formulas visible) ----
def kmeans(X, centroids, max_iter=10):
    centroids = np.array(centroids, dtype=float)
    for it in range(1, max_iter + 1):
        # distance d = sqrt((x1-c1)^2 + (x2-c2)^2) from every point to every centroid
        d = np.sqrt(((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2))
        labels = d.argmin(axis=1)                       # assign to nearest centroid
        new = np.array([X[labels == k].mean(axis=0) for k in range(len(centroids))])  # centroid = mean
        inertia = sum(((X[labels == k] - new[k]) ** 2).sum() for k in range(len(centroids)))
        print(f"Iter {it}: labels={labels}  centroids={new.round(2).tolist()}  inertia={inertia:.2f}")
        if np.allclose(new, centroids):                 # convergence: centroids stopped moving
            break
        centroids = new
    return labels, centroids, inertia

labels, cents, inertia = kmeans(X, X[[0, 5]])           # start with P1 and P6

# ---- 3. Same thing with scikit-learn ----
km = KMeans(n_clusters=2, init=X[[0, 5]], n_init=1).fit(X)
print("sklearn labels:", km.labels_, " inertia:", round(km.inertia_, 2))

# ---- 4. Choosing K: Elbow + Silhouette ----
print("\nK  inertia  silhouette")
ks, inertias = range(1, 6), []
for k in ks:
    m = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    inertias.append(m.inertia_)
    sil = silhouette_score(X, m.labels_) if k > 1 else float("nan")
    print(f"{k}  {m.inertia_:7.2f}  {sil:.3f}")

# ---- 5. Hierarchical (agglomerative) with 3 linkages ----
for method in ["single", "complete", "average"]:
    Z = linkage(X, method=method)       # each row: cluster a, cluster b, distance, size
    print(f"\n{method} linkage merges:\n", Z.round(2))
    print("cut into 2 clusters:", fcluster(Z, t=2, criterion="maxclust"))

# ---- 6. Plots ----
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
ax[0].scatter(X[:, 0], X[:, 1], c=labels); ax[0].scatter(*cents.T, marker="X", s=150, c="red")
ax[0].set_title("K-Means (K=2)")
ax[1].plot(list(ks), inertias, "o-"); ax[1].set_title("Elbow: inertia vs K"); ax[1].set_xlabel("K")
dendrogram(linkage(X, "average"), ax=ax[2], labels=["P1", "P2", "P3", "P4", "P5", "P6"])
ax[2].set_title("Dendrogram (average)")
plt.tight_layout(); plt.savefig("clustering_plots.png", dpi=120)