from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from train_test_split_data_04 import y_test
# Import the three predictions from your updated 06 file
from prediction_data_06 import pred_id3, pred_c45, pred_cart

predictions = {
    "ID3 (Entropy)": pred_id3,
    "C4.5 (Gain Ratio)": pred_c45,
    "CART (Gini)": pred_cart
}

# Loop through each model's predictions to evaluate them
for model_name, pred in predictions.items():
    print(f"\n==================== {model_name} Evaluation ====================")
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, pred))
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, pred))
