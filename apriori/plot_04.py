"""
plot_04.py  -  Runs the pipeline (load -> preprocess -> model) and plots the rules.

    python plot_04.py

Outputs (in ./outputs):
    rules_scatter.png   x = support, y = confidence, size = lift (color = lift)
    top_rules.csv       top 20 rules by lift
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from preprocess_02 import get_basket
from model_03 import MIN_SUPPORT, run, pretty

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)


def plot_rules_scatter(rules, title, path):
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(rules["support"], rules["confidence"],
                    s=rules["lift"] * 25, c=rules["lift"], cmap="viridis",
                    alpha=0.6, edgecolor="k", linewidth=0.3)
    fig.colorbar(sc, label="lift")
    ax.set_xlabel("support")
    ax.set_ylabel("confidence")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    basket, source = get_basket()
    freq, rules, sec = run(basket, MIN_SUPPORT)

    plot_rules_scatter(
        rules, f"Apriori rules (min_support={MIN_SUPPORT}, {len(rules)} rules)",
        os.path.join(OUT, "rules_scatter.png"))
    pretty(rules, 20).to_csv(os.path.join(OUT, "top_rules.csv"), index=False)

    print("=" * 62)
    print(f"APRIORI REPORT   (data source: {source})")
    print("=" * 62)
    print(f"transactions        : {len(basket)}")
    print(f"products            : {basket.shape[1]}")
    print(f"min_support         : {MIN_SUPPORT}")
    print(f"frequent itemsets   : {len(freq)}")
    print(f"rules (lift >= 1)   : {len(rules)}")
    print(f"apriori time        : {sec:.3f} s")
    print("\nTop 10 rules by lift:")
    print(pretty(rules, 10).round(3).to_string(index=False))
    print(f"\nPlots saved in: {OUT}")


if __name__ == "__main__":
    main()
