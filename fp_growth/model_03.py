"""
model_03.py  -  ONLY modelling: FP-Growth vs Apriori timing + rules. No plotting.

FP-Growth builds a compact prefix tree (FP-tree) and never generates candidate
itemsets, so it scans the data twice. Apriori generates and re-counts candidates
level by level. Both return the SAME frequent itemsets - only the speed differs.
"""
import time

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

SUPPORTS = [0.05, 0.03, 0.02, 0.015, 0.01]   # sweep for the timing comparison
MAIN_SUPPORT = 0.02                           # support used for the rules plot
REPEATS = 3                                   # take the best of N runs


def timed(fn, basket, min_support, repeats=REPEATS):
    best, result = float("inf"), None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn(basket, min_support=min_support, use_colnames=True)
        best = min(best, time.perf_counter() - t0)
    return result, best


def compare_speed(basket, supports=SUPPORTS):
    rows = []
    for s in supports:
        ap, t_ap = timed(apriori, basket, s)
        fp, t_fp = timed(fpgrowth, basket, s)
        same = set(map(frozenset, ap["itemsets"])) == set(map(frozenset, fp["itemsets"]))
        rows.append({"min_support": s, "itemsets": len(fp),
                     "apriori_s": t_ap, "fpgrowth_s": t_fp,
                     "speedup_x": t_ap / t_fp if t_fp > 0 else float("nan"),
                     "same_itemsets": same})
    return pd.DataFrame(rows)


def make_rules(basket, min_support=MAIN_SUPPORT, min_lift=1.0):
    freq = fpgrowth(basket, min_support=min_support, use_colnames=True)
    rules = association_rules(freq, num_itemsets=len(basket),
                              metric="lift", min_threshold=min_lift)
    return rules.sort_values("lift", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    from preprocess_02 import get_basket

    basket, _ = get_basket()
    print(compare_speed(basket).round(4).to_string(index=False))
