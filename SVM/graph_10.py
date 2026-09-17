import numpy as np
import matplotlib.pyplot as plt

from feature_extraction_03 import X, y
from equation_04 import model, scaler

# Job: Plot the students and the linear SVM decision boundary

X_scaled = scaler.transform(X)

# Separate the two features
x1 = X_scaled[:, 0]
x2 = X_scaled[:, 1]

# Create x-values for the decision boundary
x1_line = np.linspace(x1.min() - 1, x1.max() + 1, 100)

w1 = model.coef_[0][0]
w2 = model.coef_[0][1]
b = model.intercept_[0]

# From:
# w1*x1 + w2*x2 + b = 0
#
# Therefore:
# x2 = -(w1*x1 + b) / w2

x2_line = -(w1 * x1_line + b) / w2

plt.figure(figsize=(8, 5))

# Fail = 0, Pass = 1
plt.scatter(
    x1[y == 0],
    x2[y == 0],
    label="Fail"
)

plt.scatter(
    x1[y == 1],
    x2[y == 1],
    label="Pass"
)

plt.plot(
    x1_line,
    x2_line,
    label="SVM Hyperplane"
)

plt.xlabel("Scaled StudyHours")
plt.ylabel("Scaled Attendance")
plt.title("Linear SVM Decision Boundary")
plt.legend()
plt.grid(True)

plt.show()
