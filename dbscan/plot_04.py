"""
plot_04.py  -  Runs the whole pipeline (load -> preprocess -> model) and makes the plots.

    python plot_04.py

Outputs (in ./outputs):
    k_distance.png       k-distance graphs with the knee (suggested eps)
    eps_sweep.png        clusters / noise vs eps on blobs  (shows the 3-cluster window)
    kmeans_vs_dbscan.png side-by-side comparison on blobs and moons
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from load_01 import load_all
from preprocess_02 import scale_all
from model_03 import (MIN_SAMPLES, k_distances, knee_eps, run_dbscan, run_kmeans,
                      scan_eps, best_eps_for_k, summarize)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)


def plot_clusters(ax, X, labels, title):
    noise = labels == -1
    ax.scatter(X[~noise, 0], X[~noise, 1], c=labels[~noise], cmap="viridis", s=14)
    ax.scatter(X[noise, 0], X[noise, 1], c="red", marker="x", s=30, label="noise")
    ax.set_title(title, fontsize=10)
    if noise.any():
        ax.legend(loc="best", fontsize=8)


def main():
    data = scale_all(load_all())

    # ---- 1. k-distance graph ---------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    knee = {}
    for ax, (name, (X, _)) in zip(axes, data.items()):
        sd = k_distances(X, MIN_SAMPLES)
        eps, idx = knee_eps(sd)
        knee[name] = eps
        ax.plot(sd)
        ax.axhline(eps, color="red", ls="--", label=f"knee eps = {eps:.3f}")
        ax.scatter([idx], [eps], color="red", zorder=3)
        ax.set_title(f"k-distance graph: {name} (k={MIN_SAMPLES})")
        ax.set_xlabel("points sorted by distance")
        ax.set_ylabel(f"distance to {MIN_SAMPLES}th neighbour")
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "k_distance.png"), dpi=130)
    plt.close(fig)

    # ---- 2. eps sweep on blobs (find optimal eps for 3 clean clusters) ----
    Xb, _ = data["blobs"]
    scan = scan_eps(Xb)
    best3 = best_eps_for_k(scan, k=3)
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(scan.eps, scan.n_clusters, color="tab:blue", label="clusters")
    ax1.set_xlabel("eps")
    ax1.set_ylabel("number of clusters", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(scan.eps, scan.n_noise, color="tab:red", label="noise points")
    ax2.set_ylabel("noise points", color="tab:red")
    if best3:
        ax1.axvline(best3["eps"], color="green", ls="--")
        ax1.set_title(f"eps sweep (blobs): optimal eps for 3 clusters = {best3['eps']:.2f}")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "eps_sweep.png"), dpi=130)
    plt.close(fig)

    # ---- 3. KMeans vs DBSCAN side by side ---------------------------------
    eps_used = {"blobs": best3["eps"] if best3 else knee["blobs"], "moons": knee["moons"]}
    k_used = {"blobs": 3, "moons": 2}
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    report = []
    for row, (name, (X, _)) in enumerate(data.items()):
        km = run_kmeans(X, k_used[name])
        db = run_dbscan(X, eps_used[name], MIN_SAMPLES)
        nc, nn = summarize(db)
        plot_clusters(axes[row, 0], X, km, f"{name}: KMeans (k={k_used[name]})")
        plot_clusters(axes[row, 1], X, db,
                      f"{name}: DBSCAN (eps={eps_used[name]:.2f}, min_samples={MIN_SAMPLES})"
                      f"\nclusters={nc}, noise={nn}")
        report.append((name, eps_used[name], nc, nn))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "kmeans_vs_dbscan.png"), dpi=130)
    plt.close(fig)

    # ---- report -----------------------------------------------------------
    print("=" * 62)
    print("DBSCAN REPORT")
    print("=" * 62)
    for name in data:
        print(f"Knee eps from k-distance graph [{name}]: {knee[name]:.3f}")
    print()
    if best3:
        lo, hi = best3["stable_range"]
        print(f"Optimal eps for 3 clean clusters (blobs): {best3['eps']:.2f}")
        print(f"  working eps range (3 clusters, min noise): {lo:.2f} - {hi:.2f}")
        print(f"  noise points at optimal eps: {best3['n_noise']}")
    else:
        print("No eps in the grid gave exactly 3 clusters on blobs.")
    print()
    print(f"{'dataset':8s}{'eps':>7s}{'clusters':>10s}{'noise':>8s}")
    for name, eps, nc, nn in report:
        print(f"{name:8s}{eps:7.2f}{nc:10d}{nn:8d}")
    print(f"\nPlots saved in: {OUT}")


if __name__ == "__main__":
    main()
