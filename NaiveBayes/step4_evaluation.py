from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# Import test features from Step 2 and trained model from Step 3
from step2_feature_encoding import X_test, y_test
from step3_model_train import model

# Generate Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Calculate Performance Metrics
# predict 0 to 1 
accuracy = accuracy_score(y_test, y_pred)
# A score of 1.0 is perfect separation, while 0.5 is random guessing.
roc_auc = roc_auc_score(y_test, y_prob)
# Generates the 2 x 2 grid counting correct and incorrect predictions across both classes: True Negatives, False Positives, False Negatives, and True Positives.
cm = confusion_matrix(y_test, y_pred)

print("\n--- Step 4: Model Evaluation ---")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"ROC-AUC Score: {roc_auc:.4f}\n")
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred))