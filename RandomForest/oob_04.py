"""
Step 4 - Out-of-bag (OOB) error
-------------------------------
Each row is predicted only by trees that did NOT train on it, so OOB error is a
validation score without a separate validation split.
"""
import numpy as np
from sklearn.metrics import mean_squared_error

from model_03 import build_forest


def oob_rmse(rf, y_train) -> float:
    """OOB RMSE from rf.oob_prediction_ (one averaged OOB prediction per row)."""
    return float(np.sqrt(mean_squared_error(y_train, rf.oob_prediction_)))


def oob_vs_n_estimators(X_train, y_train, counts=(40, 80, 120, 160, 200, 300),
                        max_features=0.33):
    """Grow the same forest incrementally (warm_start) and track OOB RMSE."""
    rf = build_forest(n_estimators=counts[0], max_features=max_features,
                      warm_start=True)
    results = []
    for c in counts:
        rf.set_params(n_estimators=c)
        rf.fit(X_train, y_train)
        results.append((c, oob_rmse(rf, y_train)))
        print(f"  trees = {c:>3}  OOB RMSE = {results[-1][1]:,.0f}")
    return results


def oob_vs_max_features(X_train, y_train, options=(1, 2, 4, 6, 8, 11, 16),
                        n_estimators=100):
    """Tune m (features per node) with OOB error instead of cross-validation."""
    results = []
    for m in options:
        rf = build_forest(n_estimators=n_estimators, max_features=m)
        rf.fit(X_train, y_train)
        results.append((m, oob_rmse(rf, y_train)))
        print(f"  max_features = {m:>2}  OOB RMSE = {results[-1][1]:,.0f}")
    return results