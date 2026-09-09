from train_test_split_rule_4 import X_train, y_train, X_test, y_test
from rule_based_logics_5 import train_zero_r, train_tennis_rules

# 1. Train Rule models
zero_r_model = train_zero_r(X_train, y_train)
tennis_rule_model = train_tennis_rules(X_train, y_train)

# 2. Predict on test set
pred_zero_r = zero_r_model.predict(X_test)
pred_tennis_rules = tennis_rule_model.predict(X_test)

# 3. Print outputs
print("\n--- Actual Answers ---")
print(y_test.values)

print("\n--- ZeroR Predictions ---")
print(pred_zero_r)

print("\n--- PlayTennis Domain Rule Predictions ---")
print(pred_tennis_rules)