import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap

# Import dataset/model artifacts from Step 2, Step 3, and Step 4
from step2_feature_encoding import X_test, y_test
from step3_model_train import model
from step4_evaluation import cm, y_pred

# 1. Plot Confusion Matrix Heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Naive Bayes Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# 2. Plot Decision Boundary for Age (Index 1) and Salary (Index 2)
X_set, y_set = X_test, y_test.values
X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.02),
    np.arange(start=X_set[:, 2].min() - 1, stop=X_set[:, 2].max() + 1, step=0.02),
)

# Dummy values for Gender feature
gender_dummy = np.zeros(X1.ravel().shape)
grid_points = np.array([gender_dummy, X1.ravel(), X2.ravel()]).T

Z = model.predict(grid_points).reshape(X1.shape)

plt.figure(figsize=(8, 6))
plt.contourf(X1, X2, Z, alpha=0.3, cmap=ListedColormap(("red", "green")))

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(
        X_set[y_set == j, 1],
        X_set[y_set == j, 2],
        color=ListedColormap(("red", "green"))(i),
        label=f"Class {j}",
        edgecolors="k",
    )

plt.title("Naive Bayes Decision Boundary (Test Set)")
plt.xlabel("Age (Scaled)")
plt.ylabel("Estimated Salary (Scaled)")
plt.legend()
plt.tight_layout()
plt.savefig("decision_boundary.png")
plt.show()

print("\n--- Step 5: Visualizations Rendered & Saved ---")


# Total Correct Predictions: 61 + 32 = 93 out of 100 (93% Accuracy)
# Total Errors: 2 + 5 = 7 out of 100 (7% Error Rate)