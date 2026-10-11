"""Step 6 - Graph generation (saved as PNGs in ./plots)."""
import os
import matplotlib.pyplot as plt

OUT_DIR = "plots"


def _save(fig, name, show):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print("saved:", path)
    if show:
        plt.show()
    plt.close(fig)


def plot_oob_vs_trees(results, show=False):
    x, y = zip(*results)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, y, marker="o")
    ax.set(title="OOB error vs number of trees", xlabel="n_estimators (B)",
           ylabel="OOB RMSE")
    ax.grid(alpha=0.3)
    _save(fig, "1_oob_vs_trees.png", show)


def plot_oob_vs_max_features(results, show=False):
    x, y = zip(*results)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, y, marker="o", color="tab:green")
    ax.set(title="OOB error vs features tried per node (m)",
           xlabel="max_features (m)", ylabel="OOB RMSE")
    ax.grid(alpha=0.3)
    _save(fig, "2_oob_vs_max_features.png", show)


def plot_importance(table, show=False):
    t = table.sort_values("MDI")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    axes[0].barh(t["feature"], t["MDI"], color="tab:blue")
    axes[0].set_title("MDI importance (training)")
    axes[1].barh(t["feature"], t["Permutation"], color="tab:orange")
    axes[1].set_title("Permutation importance (test)")
    _save(fig, "3_feature_importance.png", show)


def plot_actual_vs_predicted(y_true, y_pred, show=False):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_true, y_pred, s=4, alpha=0.3)
    lim = [y_true.min(), y_true.max()]
    ax.plot(lim, lim, color="red", linestyle="--", label="perfect")
    ax.set(title="Actual vs predicted (test set)", xlabel="Actual value",
           ylabel="Predicted value")
    ax.legend()
    _save(fig, "4_actual_vs_predicted.png", show)