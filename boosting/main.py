from load_01 import load_data
from train_test_02 import prepare_data
from model_03 import get_gradient_boosting, get_xgboost, train_model
from evaluate_04 import evaluate_model, get_feature_importances
from plot_05 import plot_confusion_matrix, plot_feature_importance, plot_roc_curve


def run_pipeline():
    # 1. Load Data
    csv_file = "https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv"  # Update path as needed
    target_column = "Outcome"
    df = load_data(csv_file)

    # 2. Train-Test Split & Preprocess
    X_train, X_test, y_train, y_test = prepare_data(df, target_col=target_column)

    # 3. Train Models
    print("\nTraining Gradient Boosting...")
    gb_model = train_model(get_gradient_boosting(), X_train, y_train)

    print("Training XGBoost...")
    xgb_model = train_model(get_xgboost(), X_train, y_train)

    # 4. Evaluate Models
    gb_metrics = evaluate_model(gb_model, X_test, y_test, model_name="Gradient Boosting")
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, model_name="XGBoost")

    # 5. Extract Feature Importances
    xgb_importances = get_feature_importances(xgb_model, X_train.columns)
    print("\nTop XGBoost Feature Importances:")
    print(xgb_importances.head(5))

    # 6. Plots for XGBoost
    plot_confusion_matrix(xgb_model, X_test, y_test, title="XGBoost Confusion Matrix")
    plot_roc_curve(xgb_model, X_test, y_test, title="XGBoost ROC Curve")
    plot_feature_importance(xgb_importances, top_n=8, title="Top 8 XGBoost Features")


if __name__ == "__main__":
    run_pipeline()