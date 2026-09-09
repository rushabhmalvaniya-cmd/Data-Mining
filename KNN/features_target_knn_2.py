from load_dataset_knn_1 import df

# Features: 784 pixel columns (pixel0 to pixel783)
X = df.drop(columns=["label"])

# Target: Hand-drawn digit value (0 to 9)
y = df["label"]

print("Features Shape:", X.shape)
print("Target Shape:", y.shape)
print("\nTarget Head:")
print(y.head())