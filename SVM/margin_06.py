import numpy as np
from equation_04 import model

# Job: Calculate the SVM margin

w = model.coef_[0]

# Distance between the two margin boundaries:
# Margin = 2 / ||w||

w_norm = np.linalg.norm(w)
margin = 2 / w_norm

print("Weights (w):")
print(w)

print("\n||w||:")
print(f"{w_norm:.4f}")

print("\nSVM Margin:")
print(f"{margin:.4f}")

print("\nMargin boundaries use:")
print("w1*x1 + w2*x2 + b = +1")
print("w1*x1 + w2*x2 + b = -1")
