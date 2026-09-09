# import pandas as pd

# # Load Kaggle Digit Recognizer Image Dataset (train.csv)
# df = pd.read_csv("train.csv")

# print(f"Dataset Shape: {df.shape}")
# print("\n--- Data Information ---")
# print(df.info())

# print("\n--- First 5 Rows ---")
# print(df.head())

import pandas as pd

# 1. Load the full dataset
df_full = pd.read_csv("train.csv")

# 2. Randomly select exactly 50% of the dataset
df = df_full.sample(frac=0.2, random_state=42).reset_index(drop=True)

# 3. Print verification info
print(f"Original data size: {df_full.shape}")
print(f"New 50% data size:  {df.shape}\n")

print("--- Data Information ---")
print(df.info())

print("\n--- First 5 Rows ---")
print(df.head())