from load_dataset_rule_1 import df

# Features: Weather conditions
X = df[["outlook", "temp", "humidity", "wind"]]

# Target: PlayTennis Decision (Yes / No)
y = df["play"]

print("Features:")
print(X.head())

print("\nTarget:")
print(y.head())