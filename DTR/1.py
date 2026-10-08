import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree

# STEP 1: Dataset
X = np.array([[1], [2], [3], [4]])
y = np.array([3.2, 4.9, 7.8, 8.6])

print("Original Dataset")
print("----------------")
for i in range(len(X)):
    print(f"Point {i+1}: x = {X[i][0]}, y = {y[i]}")


# ---------------------------------------------------------
# STEP 2: Calculate possible split thresholds
# ---------------------------------------------------------

x_values = X.flatten()

thresholds = []

for i in range(len(x_values) - 1):
    threshold = (x_values[i] + x_values[i + 1]) / 2
    thresholds.append(threshold)

print("\nPossible Split Thresholds")
print("-------------------------")

for threshold in thresholds:
    print(f"x <= {threshold}")


# ---------------------------------------------------------
# STEP 3: Calculate MSE for every possible split
# ---------------------------------------------------------

print("\nSplit Calculations")
print("------------------")

best_threshold = None
best_mse = float("inf")

for threshold in thresholds:

    left_y = y[x_values <= threshold]
    right_y = y[x_values > threshold]

    left_mean = np.mean(left_y)
    right_mean = np.mean(right_y)

    # Squared errors
    left_error = np.sum((left_y - left_mean) ** 2)
    right_error = np.sum((right_y - right_mean) ** 2)

    total_error = left_error + right_error

    # Mean Squared Error
    mse = total_error / len(y)

    print(f"\nThreshold = {threshold}")

    print(f"Left values  = {left_y}")
    print(f"Right values = {right_y}")

    print(f"Left mean    = {left_mean:.2f}")
    print(f"Right mean   = {right_mean:.2f}")

    print(f"Left SSE     = {left_error:.2f}")
    print(f"Right SSE    = {right_error:.2f}")

    print(f"Total SSE    = {total_error:.2f}")
    print(f"MSE          = {mse:.2f}")

    # Find best split
    if mse < best_mse:
        best_mse = mse
        best_threshold = threshold


print("\nBest Split")
print("----------")
print(f"Best threshold = {best_threshold}")
print(f"Lowest MSE     = {best_mse:.2f}")


# ---------------------------------------------------------
# STEP 4: Train Decision Tree Regressor
# ---------------------------------------------------------

model = DecisionTreeRegressor(
    max_depth=2,
    random_state=42
)

model.fit(X, y)

print("\nDecision Tree trained successfully.")


# ---------------------------------------------------------
# STEP 5: Predictions
# ---------------------------------------------------------

predictions = model.predict(X)

print("\nPredictions")
print("-----------")

for i in range(len(X)):
    print(
        f"x = {X[i][0]} | "
        f"Actual = {y[i]} | "
        f"Predicted = {predictions[i]:.2f}"
    )


# ---------------------------------------------------------
# STEP 6: Visualize the split
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.scatter(
    X,
    y,
    s=120,
    label="Actual Data"
)

# Draw best split
plt.axvline(
    best_threshold,
    linestyle="--",
    linewidth=2,
    label=f"Best Split: x = {best_threshold}"
)

# Predictions
x_line = np.linspace(1, 4, 200).reshape(-1, 1)
y_line = model.predict(x_line)

plt.step(
    x_line.flatten(),
    y_line,
    where="post",
    linewidth=3,
    label="Decision Tree Prediction"
)

plt.xlabel("Input Feature (x)")
plt.ylabel("Target Value (y)")
plt.title("Decision Tree Regression")
plt.legend()
plt.grid(True)

plt.show()


# ---------------------------------------------------------
# STEP 7: Display the actual tree
# ---------------------------------------------------------

plt.figure(figsize=(12, 6))

plot_tree(
    model,
    feature_names=["x"],
    filled=True,
    rounded=True,
    precision=2
)

plt.title("Decision Tree Regression Structure")

plt.show()