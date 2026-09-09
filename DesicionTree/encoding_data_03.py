from features_target_02 import X, y
import pandas as pd


# Convert categorical features into numerical values
X = pd.get_dummies(X, dtype=int)

print("Encoded Features:")
print(X.head())

# -------
# The pandas function pd.get_dummies() converts categorical text data into binary numbers like 0 and 1, which is also known as one-hot encoding
# checkout for better understanding of one hot encoding: https://sharpsight.ai/wp-content/uploads/2022/04/dummy-variable_encoding-example.png