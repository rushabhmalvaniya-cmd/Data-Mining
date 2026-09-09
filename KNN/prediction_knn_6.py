from train_test_split_knn_4 import X_train, y_train, X_test, y_test
from knn_logics_5 import train_knn_euclidean, train_knn_manhattan

# 1. Train models
knn3_model = train_knn_euclidean(X_train, y_train, n_neighbors=3)
knn5_model = train_knn_euclidean(X_train, y_train, n_neighbors=5)
knn_manhattan_model = train_knn_manhattan(X_train, y_train, n_neighbors=5)

# 2. Predict on test set
pred_knn3 = knn3_model.predict(X_test)
pred_knn5 = knn5_model.predict(X_test)
pred_knn_manhattan = knn_manhattan_model.predict(X_test)

# 3. Print outputs
print("\n--- Actual Answers ---")
print(y_test.values[:10])

print("\n--- KNN (k=3, Euclidean) Predictions ---")
print(pred_knn3[:10])

print("\n--- KNN (k=5, Euclidean) Predictions ---")
print(pred_knn5[:10])

print("\n--- KNN (k=5, Manhattan) Predictions ---")
print(pred_knn_manhattan[:10])