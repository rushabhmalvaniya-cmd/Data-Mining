from feature_extraction_03 import X, y
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Job: Evaluate an SVM model
#
# This tiny teaching dataset has only 5 rows.
# To keep the example reproducible and simple, we train and evaluate
# on the same five examples.
#
# For real ML projects, use separate training and testing data.

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = SVC(kernel="linear", C=1)
model.fit(X_scaled, y)

y_pred = model.predict(X_scaled)

accuracy = accuracy_score(y, y_pred)

print("Actual Values:")
print(y.to_list())

print("\nPredicted Values:")
print(y_pred.tolist())

print("\nAccuracy:")
print(f"{accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y,
    y_pred,
    target_names=["Fail", "Pass"],
    zero_division=0
))
