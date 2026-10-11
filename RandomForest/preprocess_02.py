"""
Step 2 - Preprocessing
----------------------
Random Forest needs NO feature scaling (splits only compare order), but it
still needs numeric input and no NaNs (for this tutorial's imputation step).
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from dataset_01 import TARGET


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering (Unit II): derived ratios."""
    df = df.copy()
    df["rooms_per_household"] = df["total_rooms"] / df["households"]
    df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
    df["population_per_household"] = df["population"] / df["households"]
    return df


def prepare(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Encode, split 80/20, then impute NaNs using TRAIN medians only."""
    df = add_features(df)
    df = pd.get_dummies(df, columns=["ocean_proximity"], dtype=int)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed)

    medians = X_train.median()            # learned from train only (no leakage)
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    from dataset_01 import load_dataset
    Xtr, Xte, ytr, yte = prepare(load_dataset())
    print("Train:", Xtr.shape, "Test:", Xte.shape)
    print("Features:", list(Xtr.columns))