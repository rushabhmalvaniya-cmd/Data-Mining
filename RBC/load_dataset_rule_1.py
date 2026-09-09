import pandas as pd

# Load Kaggle Play Tennis Dataset (PlayTennis.csv)
df = pd.read_csv("play_tennis.csv")

print(f"Dataset Shape: {df.shape}")
print("\n--- Data Information ---")
print(df.info())

print("\n--- First 5 Rows ---")
print(df.head())