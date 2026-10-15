from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb


def get_gradient_boosting(learning_rate: float = 0.1, n_estimators: int = 100, max_depth: int = 3, random_state: int = 42):
    """Instantiates a Scikit-Learn Gradient Boosting Classifier."""
    return GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
    )


def get_xgboost(learning_rate: float = 0.1, n_estimators: int = 100, max_depth: int = 3, random_state: int = 42):
    """Instantiates an XGBoost Classifier."""
    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
        eval_metric="logloss",
    )


def train_model(model, X_train, y_train):
    """Trains the provided model on the training dataset."""
    model.fit(X_train, y_train)
    return model


if __name__ == "__main__":
    gb_model = get_gradient_boosting()
    xgb_model = get_xgboost()
    print("Models initialized successfully.")