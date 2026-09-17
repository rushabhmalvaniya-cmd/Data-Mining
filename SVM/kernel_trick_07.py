from feature_extraction_03 import X, y
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Job: Demonstrate the Kernel Trick
#
# Linear kernel:
# Finds a straight decision boundary.
#
# RBF kernel:
# Can create a non-linear decision boundary.

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

linear_model = SVC(kernel="linear", C=1)
linear_model.fit(X_scaled, y)

rbf_model = SVC(kernel="rbf", C=1, gamma="scale")
rbf_model.fit(X_scaled, y)

print("Linear Kernel:")
print("kernel =", linear_model.kernel)

print("\nRBF Kernel:")
print("kernel =", rbf_model.kernel)

print("\nNumber of RBF Support Vectors:")
print(len(rbf_model.support_))

print("\nImportant:")
print("The kernel trick allows SVM to handle non-linear relationships")
print("without manually creating higher-dimensional features.")
