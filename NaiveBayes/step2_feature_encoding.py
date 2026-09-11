from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Import loaded dataset from Step 1
from step1_load_dataset import df

# 1. Drop unnecessary identifiers
data = df.drop(columns=["User ID"]) if "User ID" in df.columns else df.copy()

# 2. Encode categorical column ('Gender')
label_encoder = LabelEncoder()
# male 0 , female 1
data["Gender"] = label_encoder.fit_transform(data["Gender"])

# 3. Separate features and target label
X = data.drop(columns=["Purchased"])
y = data["Purchased"]

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.25, random_state=42, stratify=y
# )

# 5. Feature Scaling (Required for Gaussian Naive Bayes)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\n--- Step 2: Features Encoded & Scaled ---")
print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")