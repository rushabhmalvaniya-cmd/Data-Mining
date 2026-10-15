import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score


def evaluate_model(model, X_test, y_test, model_name: str = "Model"):
    """Calculates and prints performance metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

    print(f"\n================ {model_name} Evaluation ================")
    print(f"Accuracy: {accuracy:.4f}")
    if roc_auc is not None:
        print(f"ROC-AUC:  {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return {
        "y_pred": y_pred,
        "y_proba": y_proba,
        "accuracy": accuracy,
        "roc_auc": roc_auc,
    }


def get_feature_importances(model, feature_names):
    """Extracts sorted feature importances into a DataFrame."""
    importances = model.feature_importances_
    return (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )