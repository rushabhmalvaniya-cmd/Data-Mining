import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from train_test_split_knn_4 import X_train, y_train, X_test, y_test
from knn_logics_5 import train_knn_euclidean
from prediction_knn_6 import knn5_model, pred_knn5

# 1. Hyperparameter Tuning Visualisation (Accuracy vs K Value)
k_range = range(1, 10)
accuracies = []

for k in k_range:
    model = train_knn_euclidean(X_train, y_train, n_neighbors=k)
    accuracies.append(model.score(X_test, y_test))

plt.figure(figsize=(10, 5))
plt.plot(k_range, accuracies, marker='o', linestyle='--', color='blue')
plt.title("KNN Image Classification Accuracy vs K Value")
plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Accuracy Score")
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Confusion Matrix Heatmap
cm = confusion_matrix(y_test, pred_knn5)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn5_model.classes_)

fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.title("KNN Image Classification Confusion Matrix")
plt.tight_layout()
plt.show()