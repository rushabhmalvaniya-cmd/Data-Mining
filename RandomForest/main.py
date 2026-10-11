"""Run the whole pipeline:  python main.py"""
from dataset_01 import load_dataset, explore, KAGGLE_PAGE, DATA_URL
from preprocess_02 import prepare
from model_03 import build_forest, bootstrap_demo, evaluate
from oob_04 import oob_rmse, oob_vs_n_estimators, oob_vs_max_features
from importance_05 import importance_table
import plot_06 as plots

SHOW = False   # True = also open the plot windows

print("Dataset:", KAGGLE_PAGE, "\nLoaded from:", DATA_URL, "\n")
df = load_dataset()
explore(df)

X_train, X_test, y_train, y_test = prepare(df)
print("\nTrain:", X_train.shape, " Test:", X_test.shape)
bootstrap_demo(len(X_train))

print("\n[OOB vs number of trees]")
plots.plot_oob_vs_trees(oob_vs_n_estimators(X_train, y_train), SHOW)

print("\n[OOB vs max_features]")
res = oob_vs_max_features(X_train, y_train)
plots.plot_oob_vs_max_features(res, SHOW)
best_m = min(res, key=lambda r: r[1])[0]
print("Best m by OOB:", best_m)

print("\n[Final forest]")
rf = build_forest(n_estimators=200, max_features=best_m).fit(X_train, y_train)
print(f"OOB R2 = {rf.oob_score_:.3f}   OOB RMSE = {oob_rmse(rf, y_train):,.0f}")
pred, _, _ = evaluate(rf, X_test, y_test)

table = importance_table(rf, X_test, y_test)
print("\n", table.round(4))
plots.plot_importance(table, SHOW)
plots.plot_actual_vs_predicted(y_test, pred, SHOW)