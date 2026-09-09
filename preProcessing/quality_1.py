import pandas as pd

def inspect_data_quality(df: pd.DataFrame):
    """Prints a dataset quality summary including data types, missing values, and duplicates."""
    print("=== Data Quality Summary ===")
    print(f"Shape: {df.shape}")
    print("\n--- Missing Values ---")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    print("\n--- Duplicate Rows ---")
    print(f"Duplicates count: {df.duplicated().sum()}")
    print("\n--- Numerical Statistics ---")
    print(df.describe().T)

if __name__ == "__main__":
    df_sample = pd.read_csv("healthcare-dataset-stroke-data.csv").sample(frac=0.5, random_state=42)
    inspect_data_quality(df_sample)