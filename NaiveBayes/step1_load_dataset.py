import pandas as pd

# 1. Load Dataset
df = pd.read_csv("Social_Network_Ads.csv")

print("--- Step 1: Data Loaded ---")
print(f"Dataset Shape: {df.shape}")
print(df.head())