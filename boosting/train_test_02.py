import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def prepare_data(
    df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42
):
    """Cleans missing values, performs train-test split, and scales features."""
    df_clean = df.dropna().copy()

    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    return X_train_scaled, X_test_scaled, y_train, y_test


if __name__ == "__main__":
    from load_01 import load_data

    df = load_data("diabetes.csv")
    X_train, X_test, y_train, y_test = prepare_data(df, target_col="Outcome")
    print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")