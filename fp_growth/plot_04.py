"""
plot_04.py  -  Runs the pipeline (load -> preprocess -> model) and makes the plots.

    python plot_04.py

Outputs (in ./outputs):
    speed_comparison.png  Apriori vs FP-Growth execution time for several min_support values
    rules_scatter.png     x = support, y = confidence, size = lift  (rules from FP-Growth)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from preprocess_02 import get_basket
from model_03 import MAIN_SUPPORT, compare_speed, make_rules

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)


def main():
    basket, source = get_basket()

    # ---- speed comparison -------------------------------------------------
    speed = compare_speed(basket)
    x = np.arange(len(speed))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w / 2, speed.apriori_s, w, label="Apriori", color="tab:orange")
    ax.bar(x + w / 2, speed.fpgrowth_s, w, label="FP-Growth", color="tab:blue")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in speed.min_support])
    ax.set_xlabel("min_support")
    ax.set_ylabel("execution time (s)")
    ax.set_title("Apriori vs FP-Growth execution time")
    ax.legend()
    for i, r in speed.iterrows():
        ax.text(i, max(r.apriori_s, r.fpgrowth_s), f"{r.speedup_x:.1f}x",
                ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "speed_comparison.png"), dpi=130)
    plt.close(fig)

    # ---- rules scatter ----------------------------------------------------
    rules = make_rules(basket, MAIN_SUPPORT)
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(rules["support"], rules["confidence"],
                    s=rules["lift"] * 25, c=rules["lift"], cmap="viridis",
                    alpha=0.6, edgecolor="k", linewidth=0.3)
    fig.colorbar(sc, label="lift")
    ax.set_xlabel("support")
    ax.set_ylabel("confidence")
    ax.set_title(f"FP-Growth rules (min_support={MAIN_SUPPORT}, {len(rules)} rules)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "rules_scatter.png"), dpi=130)
    plt.close(fig)

    # ---- report -----------------------------------------------------------
    print("=" * 70)
    print(f"FP-GROWTH vs APRIORI   (data source: {source}, {len(basket)} transactions, "
          f"{basket.shape[1]} products)")
    print("=" * 70)
    print(speed.round(4).to_string(index=False))
    print(f"\nSame itemsets from both algorithms: {bool(speed.same_itemsets.all())}")
    print(f"Average speed difference : FP-Growth is {speed.speedup_x.mean():.1f}x faster")
    print(f"Largest speed difference : {speed.speedup_x.max():.1f}x "
          f"(at min_support={speed.loc[speed.speedup_x.idxmax(), 'min_support']})")
    print(f"\nRules at min_support={MAIN_SUPPORT}: {len(rules)}")
    print(f"Plots saved in: {OUT}")


if __name__ == "__main__":
    main()
