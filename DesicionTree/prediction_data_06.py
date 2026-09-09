from train_test_split_data_04 import X_train, y_train, X_test, y_test
from decision_tree_logics_05 import train_id3, train_c45, train_cart

# 1. Train the three different models using training data
id3_model = train_id3(X_train, y_train)
c45_model = train_c45(X_train, y_train)
cart_model = train_cart(X_train, y_train)

# 2. Make predictions on the test set with each model
pred_id3 = id3_model.predict(X_test)
pred_c45 = c45_model.predict(X_test)
pred_cart = cart_model.predict(X_test)

# 3. Print out the comparisons
print("\n--- Actual Answers ---")
print(y_test.values)

print("\n--- ID3 Predictions ---")
print(pred_id3)

print("\n--- C4.5 Predictions ---")
print(pred_c45)

print("\n--- CART Predictions ---")
print(pred_cart)
