from data_cleaning_3 import df_clean
from preprocessing_functions_4 import (
    handle_outliers_iqr,
    bin_feature,
    encode_categorical,
    normalize_data
)

num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
cat_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
if "Churn" in cat_cols:
    cat_cols.remove("Churn")

# 1. Apply IQR Capping on numerical features
df_transformed = handle_outliers_iqr(df_clean, columns=num_cols, action='cap')

# 2. Apply Binning on 'tenure'
bins = [0, 12, 24, 48, 72]
labels = ['0-1 Year', '1-2 Years', '2-4 Years', '4-6 Years']
df_transformed = bin_feature(df_transformed, column='tenure', bins=bins, labels=labels)

# Add newly created bin column to categorical list
cat_cols.append('tenure_Group')

# 3. Apply Categorical Encoding
df_transformed = encode_categorical(df_transformed, categorical_cols=cat_cols, method='onehot')

# 4. Encode Binary Target Column
if "Churn" in df_transformed.columns:
    df_transformed["Churn"] = (df_transformed["Churn"] == "Yes").astype(int)

# 5. Apply Normalization on numerical features
df_transformed, scaler_obj = normalize_data(df_transformed, columns=num_cols, method='standard')

print("\nTransformed Dataset Shape:", df_transformed.shape)
print(df_transformed.head())