from sklearn.model_selection import train_test_split
from feature_transformation_5 import df_transformed

# Separate features (X) and target (y)
X = df_transformed.drop(columns=["Churn"])
y = df_transformed["Churn"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train class distribution:\n", y_train.value_counts(normalize=True))