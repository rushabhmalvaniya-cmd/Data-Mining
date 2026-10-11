"""
Step 5 - Feature importance
---------------------------
MDI         : impurity (variance) reduction, computed during training
Permutation : drop in score when one feature is shuffled (on held-out data)
"""
import pandas as pd
from sklearn.inspection import permutation_importance


def importance_table(rf, X_test, y_test, n_repeats=5, seed=42) -> pd.DataFrame:
    perm = permutation_importance(rf, X_test, y_test, n_repeats=n_repeats,
                                  random_state=seed, n_jobs=-1)
    table = pd.DataFrame({
        "feature": X_test.columns,
        "MDI": rf.feature_importances_,
        "Permutation": perm.importances_mean,
    }).sort_values("MDI", ascending=False).reset_index(drop=True)
    return table