# import pandas as pd

# df = pd.read_csv("ubr.csv")

# print(df)
# # ------
# print(df.head())
# print(df.info())
# print(df.describe())


# -------------
import pandas as pd

# 1. Load the full dataset
df_full = pd.read_csv("ubr.csv")

# 2. Randomly select exactly 50% of the dataset
# frac=0.5 means 50%, random_state=42 keeps it consistent
df = df_full.sample(frac=0.5, random_state=42).reset_index(drop=True)

# 3. Print verification info
print(f"Original data size: {df_full.shape}")
print(f"New 50% data size:  {df.shape}\n")

print("--- Data Information ---")
print(df.info())

print("\n--- Summary Statistics ---")
print(df.describe())
