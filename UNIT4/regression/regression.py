import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, r2_score

# Import Scikit-Learn Algorithms
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# 1. Load Kaggle House Prices Dataset directly via raw URL
url = "https://raw.githubusercontent.com/sidhantagar/Kaggle-House-Prices/master/train.csv"
df = pd.read_csv(url)

# Select key numerical features
features = ['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea']
target = 'SalePrice'

# Filter dataset
df = df[features + [target]].dropna()
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features (Required for Linear/SVR; harmless for Trees)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==============================================================================
# STEP 2: UNCOMMENT EXACTLY ONE MODEL TO TEST IT
# ==============================================================================
model = LinearRegression()
# model = Ridge(alpha=10.0)
# model = Lasso(alpha=100.0)
# model = SVR(kernel="rbf", C=100000, epsilon=0.1)
# model = DecisionTreeRegressor(max_depth=5, random_state=42)
# model = RandomForestRegressor(n_estimators=100, random_state=42)
# model = GradientBoostingRegressor(n_estimators=100, random_state=42)
# ==============================================================================

# 3. Fit, Predict & Evaluate (Identical code for every algorithm)
model.fit(X_train, y_train)
preds = model.predict(X_test)

print(f"Active Algorithm : {model.__class__.__name__}")
print(f"RMSE             : ${root_mean_squared_error(y_test, preds):,.2f}")
print(f"R2 Score         : {r2_score(y_test, preds):.4f}")