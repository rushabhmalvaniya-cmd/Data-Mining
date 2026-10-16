from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt

# REMOVED: matplotlib.use("Agg")  <-- Allows plots to pop up / render inline

# 1. Load dataset (Public GitHub mirror of Kaggle Mall Customers CSV)
URL = "https://raw.githubusercontent.com/PacktPublishing/Data-Science-for-Marketing-Analytics-Second-Edition/master/Chapter03/Datasets/Mall_Customers.csv"

try:
    df = pd.read_csv(URL)
except Exception:
    df = pd.read_csv("Mall_Customers.csv")  # Fallback to local file if offline

X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# 2. Step 1: Elbow method to choose K
inertias = [
    KMeans(n_clusters=k, n_init=10, random_state=42).fit(X).inertia_
    for k in range(1, 11)
]

plt.figure(figsize=(8, 4.5))
plt.plot(range(1, 11), inertias, "o-")
plt.xlabel("K")
plt.ylabel("Inertia")
plt.title("Elbow Method (bend at K = 5)")
plt.savefig("2a_kmeans_elbow.png", dpi=150, bbox_inches="tight")
plt.show()  # Display plot interactively

# 3. Step 2 & 3: Fit K-Means (K = 5) and plot clusters
model = KMeans(n_clusters=5, n_init=10, random_state=42)
df["Cluster"] = model.fit_predict(X)

plt.figure(figsize=(8, 5))
plt.scatter(
    X.iloc[:, 0], X.iloc[:, 1], c=df["Cluster"], cmap="viridis", edgecolor="k"
)
plt.scatter(
    *model.cluster_centers_.T,
    c="red",
    marker="X",
    s=250,
    edgecolor="k",
    label="Centres"
)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("K-Means (K = 5)")
plt.legend()
plt.savefig("2b_kmeans_clusters.png", dpi=150, bbox_inches="tight")
plt.show()  # Display plot interactively

# 4. Print cluster summary statistics
print(
    df.groupby("Cluster")[["Annual Income (k$)", "Spending Score (1-100)"]]
    .mean()
    .round(1)
)