from feature_extraction_03 import X, y
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Job:
# 1. Scale the features
# 2. Train a linear SVM
# 3. Generate the hyperplane equation

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = SVC(kernel="linear", C=1)
model.fit(X_scaled, y)

# Linear SVM equation:
# w1*x1 + w2*x2 + b = 0

w1 = model.coef_[0][0]
w2 = model.coef_[0][1]
b = model.intercept_[0]

print("Linear SVM trained successfully.")

print("\nHyperplane Equation:")
print(f"{w1:.4f} * x1 + {w2:.4f} * x2 + {b:.4f} = 0")

print("\nWhere:")
print("x1 = scaled StudyHours")
print("x2 = scaled Attendance")
print("w1, w2 = weights")
print("b = bias")
