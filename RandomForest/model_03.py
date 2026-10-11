"""
Step 3 - Build the forest (construction, feature subset per node)
-----------------------------------------------------------------
bootstrap=True      -> each tree trains on a bootstrap sample (~63.2% unique rows)
max_features        -> m features tried at EVERY node (re-drawn per node)
oob_score=True      -> keeps OOB predictions for the free validation estimate
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


def build_forest(n_estimators=100, max_features=0.33, max_depth=None,
                 min_samples_leaf=1, warm_start=False, seed=42):
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_features=max_features,   # float = fraction of p, int = exact m
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        bootstrap=True,
        oob_score=True,
        warm_start=warm_start,
        n_jobs=-1,
        random_state=seed,
    )


def bootstrap_demo(n_rows: int, seed: int = 0) -> None:
    """Show that one bootstrap sample keeps ~63.2% unique rows, ~36.8% are OOB."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_rows, size=n_rows)       # draw with replacement
    unique = len(np.unique(idx)) / n_rows
    print(f"Bootstrap sample: {unique:.1%} unique rows in-bag, "
          f"{1 - unique:.1%} out-of-bag (theory: 63.2% / 36.8%)")


def evaluate(rf, X, y, label="Test"):
    pred = rf.predict(X)
    rmse = np.sqrt(mean_squared_error(y, pred))
    r2 = r2_score(y, pred)
    print(f"{label}: RMSE = {rmse:,.0f}  R2 = {r2:.3f}")
    return pred, rmse, r2