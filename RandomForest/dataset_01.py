"""
Step 1 - Load the dataset
-------------------------
Kaggle: California Housing Prices (1990 census, 20,640 districts, 10 columns)
Page  : https://www.kaggle.com/datasets/camnugent/california-housing-prices

Kaggle downloads need a login, so we read the same CSV straight from a public
GitHub mirror (Aurelien Geron's Hands-On ML repo, the source of this dataset).
Optional: with the `kagglehub` package and a Kaggle account you can instead use
    kagglehub.dataset_download("camnugent/california-housing-prices")
"""
import pandas as pd

KAGGLE_PAGE = "https://www.kaggle.com/datasets/camnugent/california-housing-prices"
DATA_URL = ("https://raw.githubusercontent.com/ageron/handson-ml2/"
            "master/datasets/housing/housing.csv")
TARGET = "median_house_value"


def load_dataset(url: str = DATA_URL) -> pd.DataFrame:
    """Read the CSV from the live URL into a DataFrame."""
    return pd.read_csv(url)


def explore(df: pd.DataFrame) -> None:
    """Quick EDA (Unit I): shape, head, dtypes, missing values, statistics."""
    print("Shape:", df.shape)
    print("\n--- head() ---\n", df.head())
    print("\n--- info() ---")
    df.info()
    print("\n--- missing values ---\n", df.isna().sum()[df.isna().sum() > 0])
    print("\n--- describe() ---\n", df.describe().T)
    print("\n--- ocean_proximity counts ---\n", df["ocean_proximity"].value_counts())


if __name__ == "__main__":
    data = load_dataset()
    explore(data)