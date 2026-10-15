import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    """Loads a CSV file into a Pandas DataFrame."""
    try:
        df = pd.read_csv(filepath)
        print(f"Data successfully loaded from '{filepath}'. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading file '{filepath}': {e}")
        raise


if __name__ == "__main__":
    # Example usage
    df = load_data("diabetes.csv")
    print(df.head())