from features_target_knn_2 import X, y

# Scale pixel values from range [0, 255] to range [0.0, 1.0] for distance calculations
X_scaled = X / 255.0

print("Scaled Features Head:")
print(X_scaled.head())