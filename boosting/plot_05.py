import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


def plot_confusion_matrix(model, X_test, y_test, title: str = "Confusion Matrix"):
    """Plots and displays the confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap="Blues", ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_roc_curve(model, X_test, y_test, title: str = "ROC Curve"):
    """Plots and displays the ROC Curve."""
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.set_title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(importance_df, top_n: int = 10, title: str = "Feature Importance"):
    """Plots top features based on importance scores."""
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=importance_df.head(top_n),
        x="Importance",
        y="Feature",
        palette="viridis",
        hue="Feature",
        legend=False,
    )
    plt.title(title)
    plt.tight_layout()
    plt.show()