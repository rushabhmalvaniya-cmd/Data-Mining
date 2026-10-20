"""
plot_04.py  -  Runs the pipeline (load -> preprocess -> model) and makes the plots.

    python plot_04.py

Outputs (in ./outputs):
    k_sweep_metrics.png   Davies-Bouldin and Calinski-Harabasz vs k (KMeans, blobs)
    algorithm_metrics.png metric comparison KMeans / Agglomerative / DBSCAN
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from load_01 import load_all
from preprocess_02 import scale_all
from model_03 import kmeans_k_sweep, compare_algorithms

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)

# k for KMeans/Agglomerative and eps for DBSCAN per dataset
SETTINGS = {"blobs": {"k": 4, "eps": 0.30}, "moons": {"k": 2, "eps": 0.30}}


def main():
    data = scale_all(load_all())

    # ---- 1. metric vs k ---------------------------------------------------
    Xb, _ = data["blobs"]
    sweep = kmeans_k_sweep(Xb)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col, title, best in [
        (axes[0], "davies_bouldin", "Davies-Bouldin (lower is better)", sweep.davies_bouldin.idxmin()),
        (axes[1], "calinski_harabasz", "Calinski-Harabasz (higher is better)", sweep.calinski_harabasz.idxmax()),
        (axes[2], "silhouette", "Silhouette (higher is better)", sweep.silhouette.idxmax()),
    ]:
        ax.plot(sweep.k, sweep[col], marker="o")
        ax.axvline(sweep.k[best], color="green", ls="--", label=f"best k = {sweep.k[best]}")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("k")
        ax.legend()
    fig.suptitle("KMeans on blobs: metric vs number of clusters")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "k_sweep_metrics.png"), dpi=130)
    plt.close(fig)

    # ---- 2. algorithm comparison -----------------------------------------
    tables = []
    for name, (X, _) in data.items():
        s = SETTINGS[name]
        df, _ = compare_algorithms(X, k=s["k"], eps=s["eps"])
        df.insert(0, "dataset", name)
        tables.append(df)
    result = pd.concat(tables, ignore_index=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for row, name in enumerate(data):
        sub = result[result.dataset == name]
        for col, (metric, title) in enumerate([
            ("davies_bouldin", "Davies-Bouldin (lower better)"),
            ("calinski_harabasz", "Calinski-Harabasz (higher better)"),
            ("silhouette", "Silhouette (higher better)"),
        ]):
            ax = axes[row, col]
            ax.bar(sub.algorithm.str.replace("(", "\n(", regex=False), sub[metric],
                   color=["tab:blue", "tab:orange", "tab:green"])
            ax.set_title(f"{name}: {title}", fontsize=9)
            ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "algorithm_metrics.png"), dpi=130)
    plt.close(fig)

    # ---- report -----------------------------------------------------------
    pd.set_option("display.width", 140)
    print("=" * 70)
    print("KMeans k sweep on blobs")
    print("=" * 70)
    print(sweep.round(3).to_string(index=False))
    print(f"\nBest k by Davies-Bouldin     : {sweep.k[sweep.davies_bouldin.idxmin()]}")
    print(f"Best k by Calinski-Harabasz  : {sweep.k[sweep.calinski_harabasz.idxmax()]}")
    print("\n" + "=" * 70)
    print("Algorithm comparison")
    print("=" * 70)
    print(result.round(3).to_string(index=False))
    print("\nNOTE: on moons, centroid-based metrics (DB, CH) can favour KMeans even")
    print("though DBSCAN recovers the true shape. Always look at the plot as well.")
    print(f"\nPlots saved in: {OUT}")


if __name__ == "__main__":
    main()
