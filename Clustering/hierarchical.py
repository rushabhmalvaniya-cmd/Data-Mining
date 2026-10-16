from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import matplotlib.pyplot as plt


# 1. Load dataset (Public GitHub mirror of Kaggle Mall Customers CSV)
URL = "https://raw.githubusercontent.com/PacktPublishing/Data-Science-for-Marketing-Analytics-Second-Edition/master/Chapter03/Datasets/Mall_Customers.csv"

try:
    df = pd.read_csv(URL)
except Exception:
    df = pd.read_csv("Mall_Customers.csv")  # Fallback to local file if offline

# Extract features for clustering
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# 2. Fit Agglomerative Hierarchical Clustering (5 clusters)
model = AgglomerativeClustering(n_clusters=5, linkage="ward")
df["Cluster"] = model.fit_predict(X)

# 3. Plot 1: Cluster Scatter Plot
plt.figure(figsize=(8, 5))
plt.scatter(
    X.iloc[:, 0], X.iloc[:, 1], c=df["Cluster"], cmap="viridis", edgecolor="k"
)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Hierarchical Clustering (5 clusters)")
plt.savefig("3a_hier_clusters.png", dpi=150, bbox_inches="tight")
plt.show()  # Display plot interactively

# 4. Plot 2: Dendrogram (Merge Tree)
plt.figure(figsize=(10, 5))
dendrogram(linkage(X, method="ward"), truncate_mode="lastp", p=15)
plt.xlabel("Cluster (number of points)")
plt.ylabel("Merge distance")
plt.title("Dendrogram")
plt.savefig("3b_hier_dendrogram.png", dpi=150, bbox_inches="tight")
plt.show()  # Display plot interactively