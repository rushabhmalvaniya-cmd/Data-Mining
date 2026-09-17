from equation_04 import model, X_scaled, y

# Job: Find the Support Vectors

support_vectors = model.support_vectors_

print("Support Vectors:")
print(support_vectors)

print("\nNumber of Support Vectors:")
print(len(support_vectors))

print("\nSupport Vector Indices:")
print(model.support_)

print("\nSupport Vector Classes:")
print(y.iloc[model.support_].to_list())

# Note:
# SVM decides the exact support vectors after training.
# Do not assume them only by looking at the raw dataset.
