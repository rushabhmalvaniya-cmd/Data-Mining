from sklearn.metrics import mean_squared_error, r2_score

def evaluate_and_predict(model, X_test_scaled, y_test, scaler_X, scaler_y, unseen_val=6.5):
    # Test evaluation
    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Unseen data prediction
    unseen_scaled = scaler_X.transform([[unseen_val]])
    pred_scaled = model.predict(unseen_scaled)
    pred_salary = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]

    return mse, r2, pred_salary