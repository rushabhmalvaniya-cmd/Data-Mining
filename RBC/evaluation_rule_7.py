from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from train_test_split_rule_4 import y_test
from prediction_rule_6 import pred_zero_r, pred_tennis_rules

predictions = {
    "ZeroR (Majority Rule)": pred_zero_r,
    "PlayTennis Domain Rule Classifier": pred_tennis_rules
}

for model_name, pred in predictions.items():
    print(f"\n==================== {model_name} Evaluation ====================")
    accuracy = accuracy_score(y_test, pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, pred, zero_division=0))