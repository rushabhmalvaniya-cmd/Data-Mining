import pandas as pd
import numpy as np
from load_1 import df

df_clean = df.copy()

# 1. Drop identifier column
if "customerID" in df_clean.columns:
    df_clean.drop(columns=["customerID"], inplace=True)

# 2. Fix data type: Convert 'TotalCharges' from string/object to float
df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"].str.strip(), errors="coerce")

# 3. Handle missing values using Median Imputation
missing_count = df_clean["TotalCharges"].isnull().sum()
if missing_count > 0:
    median_val = df_clean["TotalCharges"].median()
    df_clean["TotalCharges"].fillna(median_val, inplace=True)
    print(f"Imputed {missing_count} missing values in 'TotalCharges' with median ({median_val:.2f})")

# 4. Drop duplicate rows if present
initial_len = len(df_clean)
df_clean.drop_duplicates(inplace=True)
print(f"Removed {initial_len - len(df_clean)} duplicate rows.")

print("\nCleaned Data Summary:")
print(df_clean.info())