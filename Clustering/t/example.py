import matplotlib.pyplot as plt

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

# 1. Load images (1797 images, 8x8 flattened to 64 pixels)
digits = load_digits()
X = digits.data

# 2. Fit K-Means
model = KMeans(n_clusters=10, n_init=10, random_state=42)
labels = model.fit_predict(X)

# 3. Plot 1: Cluster centers
fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))
for k, ax in enumerate(axes.ravel()):
    ax.imshow(model.cluster_centers_[k].reshape(8, 8), cmap="gray_r")
    ax.set_title(f"Cluster {k}")
    ax.axis("off")
fig.suptitle("K-Means cluster centres (average image per cluster)")
plt.savefig("4a_centres.png", dpi=150, bbox_inches="tight")
plt.show()  # Display plot interactively

# 4. Plot 2: PCA 2D Projection
pca = PCA(n_components=2).fit(X)
P = pca.transform(X)
C = pca.transform(model.cluster_centers_)

plt.figure(figsize=(8, 7))
plt.scatter(P[:, 0], P[:, 1], c=labels, cmap="tab10", s=15)
plt.scatter(
    C[:, 0],
    C[:, 1],
    c="red",
    marker="X",
    s=250,
    edgecolor="k",
    label="Centres",
)
for k, (x, y) in enumerate(C):
    plt.text(x, y + 1, str(k), fontsize=12, weight="bold", ha="center")
plt.title("K-Means on digit images (2D view)")
plt.legend()
plt.savefig("4b_scatter.png", dpi=150, bbox_inches="tight")
plt.show()  # Display plot interactively