import pandas as pd

# 1. Load full dataset
df_full = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# 2. Randomly select 50% of dataset
df = df_full.sample(frac=0.5, random_state=42).reset_index(drop=True)

# 3. Print dataset verification
print(f"Original Dataset Size: {df_full.shape}")
print(f"50% Sampled Dataset Size: {df.shape}\n")

print("--- First 5 Rows ---")
print(df.head())