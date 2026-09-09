import pandas as pd
from data_cleaning_3 import df_clean
from preprocessing_functions_4 import apply_pca

num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

# Fill missing values in numerical columns with their median before PCA
df_clean[num_cols] = df_clean[num_cols].fillna(df_clean[num_cols].median())

# Apply PCA dimensionality reduction on numerical feature set
df_pca_components, pca_model = apply_pca(df_clean, numerical_cols=num_cols, n_components=2)

print("\n--- PCA Reduced DataFrame Head ---")
print(df_pca_components.head())