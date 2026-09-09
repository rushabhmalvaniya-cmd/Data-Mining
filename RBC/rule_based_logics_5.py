import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

class ZeroRClassifier(BaseEstimator, ClassifierMixin):
    """ZeroR Classifier: Baseline rule predicting the majority class."""
    def __init__(self):
        self.majority_class_ = None

    def fit(self, X, y):
        self.majority_class_ = pd.Series(y).mode()[0]
        self.classes_ = np.unique(y)
        print(f"ZeroR Model trained! Majority Class: '{self.majority_class_}'")
        return self

    def predict(self, X):
        return np.full(shape=(len(X),), fill_value=self.majority_class_)

class PlayTennisRuleClassifier(BaseEstimator, ClassifierMixin):
    """Domain Rule Classifier implementing standard decision rules for PlayTennis."""
    def __init__(self):
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        print("PlayTennis Rule-Based Model trained successfully!")
        return self

    def predict(self, X):
        X_df = pd.DataFrame(X)
        preds = []
        for _, row in X_df.iterrows():
            outlook = str(row.get('Outlook')).strip().title()
            humidity = str(row.get('Humidity')).strip().title()
            wind = str(row.get('Wind')).strip().title()
            
            # Rule 1: IF Outlook == 'Overcast' -> Predict 'Yes'
            if outlook == 'Overcast':
                preds.append('Yes')
            # Rule 2: IF Outlook == 'Sunny' AND Humidity == 'Normal' -> 'Yes' ELSE 'No'
            elif outlook == 'Sunny':
                if humidity == 'Normal':
                    preds.append('Yes')
                else:
                    preds.append('No')
            # Rule 3: IF Outlook == 'Rain' AND Wind in ('Weak', 'False') -> 'Yes' ELSE 'No'
            elif outlook in ('Rain', 'Rainy'):
                if wind in ('Weak', 'False'):
                    preds.append('Yes')
                else:
                    preds.append('No')
            # Fallback Rule
            else:
                preds.append('Yes')
        return np.array(preds)

def train_zero_r(X_tr, y_tr):
    model = ZeroRClassifier()
    model.fit(X_tr, y_tr)
    return model

def train_tennis_rules(X_tr, y_tr):
    model = PlayTennisRuleClassifier()
    model.fit(X_tr, y_tr)
    return model