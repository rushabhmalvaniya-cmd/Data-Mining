from step1_load_data import load_dataset
from step2_split_data import split_dataset
from step3_scale_data import scale_dataset
from step4_train_model import train_svr
from step5_evaluate import evaluate_and_predict
from step6_plot import plot_svr

# Hyperparameters
E = 0.2
C_VAL = 10.0

# Pipeline execution
X, y = load_dataset()
X_train, X_test, y_train, y_test = split_dataset(X, y)
X_tr_s, X_te_s, y_tr_s, y_te_s, scaler_X, scaler_y = scale_dataset(X_train, X_test, y_train, y_test)

model = train_svr(X_tr_s, y_tr_s, C=C_VAL, epsilon=E)
mse, r2, pred = evaluate_and_predict(model, X_te_s, y_test, scaler_X, scaler_y, unseen_val=6.5)

print(f"MSE: {mse:,.2f}")
print(f"R2 Score: {r2:.4f}")
print(f"Predicted Salary (6.5 yrs): ${pred:,.2f}")

plot_svr(model, X_tr_s, y_tr_s, epsilon=E)