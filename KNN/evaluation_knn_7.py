from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from train_test_split_knn_4 import y_test
from prediction_knn_6 import pred_knn3, pred_knn5, pred_knn_manhattan

predictions = {
    "KNN (k=3, Euclidean)": pred_knn3,
    # "KNN (k=5, Euclidean)": pred_knn5,
    # "KNN (k=5, Manhattan)": pred_knn_manhattan
}

for model_name, pred in predictions.items():
    print(f"\n==================== {model_name} Evaluation ====================")
    accuracy = accuracy_score(y_test, pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, pred, zero_division=0))