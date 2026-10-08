from sklearn.tree import DecisionTreeRegressor, export_text

X, y = [[1], [2], [3], [4]], [3.2, 4.9, 7.8, 8.6]

model = DecisionTreeRegressor(
    criterion="squared_error",   # variance reduction
    max_depth=1,
    min_samples_leaf=1,
    random_state=42
)
model.fit(X, y)

print(export_text(model, feature_names=["x"]))   # shows the rule x <= 2.5
print(model.predict([[2.5]]))                    # [4.05]
print(model.score(X, y))                         # R² on training data
