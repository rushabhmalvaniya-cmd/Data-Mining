import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

# Import Scikit-Learn Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

# 1. Load Kaggle Telco Customer Churn Dataset
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)

# Clean numeric target and features
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()

features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
X = df[features]
y = df['Churn'].map({'Yes': 1, 'No': 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==============================================================================
# STEP 2: UNCOMMENT EXACTLY ONE MODEL TO TEST IT
# ==============================================================================
# model = LogisticRegression()
# model = SVC(kernel="rbf", C=1.0)
# model = DecisionTreeClassifier(max_depth=5, random_state=42)
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model = AdaBoostClassifier(n_estimators=100, random_state=42)
model = GradientBoostingClassifier(n_estimators=100, random_state=42)
# ==============================================================================

# 3. Fit, Predict & Evaluate (Identical code for every algorithm)
model.fit(X_train, y_train)
preds = model.predict(X_test)

print(f"Active Algorithm : {model.__class__.__name__}")
print(f"Accuracy         : {accuracy_score(y_test, preds):.4f}")
print(f"F1 Score         : {f1_score(y_test, preds):.4f}")