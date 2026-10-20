"""
model_03.py  -  ONLY modelling: apriori() + association_rules(). No plotting.
"""
import time

from mlxtend.frequent_patterns import apriori, association_rules

MIN_SUPPORT = 0.02
MIN_LIFT = 1.0


def mine_frequent_itemsets(basket, min_support=MIN_SUPPORT, max_len=None):
    return apriori(basket, min_support=min_support, use_colnames=True, max_len=max_len)


def make_rules(freq, n_transactions, metric="lift", min_threshold=MIN_LIFT):
    if freq.empty:
        return freq
    rules = association_rules(freq, num_itemsets=n_transactions,
                              metric=metric, min_threshold=min_threshold)
    return rules.sort_values("lift", ascending=False).reset_index(drop=True)


def pretty(rules, n=10):
    """Readable top-n rules."""
    out = rules.head(n).copy()
    out["antecedents"] = out["antecedents"].apply(lambda s: ", ".join(sorted(s)))
    out["consequents"] = out["consequents"].apply(lambda s: ", ".join(sorted(s)))
    return out[["antecedents", "consequents", "support", "confidence", "lift"]]


def run(basket, min_support=MIN_SUPPORT):
    t0 = time.perf_counter()
    freq = mine_frequent_itemsets(basket, min_support)
    elapsed = time.perf_counter() - t0
    rules = make_rules(freq, len(basket))
    return freq, rules, elapsed


if __name__ == "__main__":
    from preprocess_02 import get_basket

    basket, _ = get_basket()
    freq, rules, sec = run(basket)
    print(f"Apriori: {len(freq)} frequent itemsets, {len(rules)} rules, {sec:.3f}s")
    print(pretty(rules, 5).round(3).to_string(index=False))
