from sklearn.tree import DecisionTreeClassifier
from train_test_split_data_04 import X_train, X_test, y_train, y_test

def train_id3(X_tr, y_tr):
    """Trains a decision tree using Information Gain (Entropy) with balanced weights."""
    # Added class_weight='balanced' to handle class imbalance
    model = DecisionTreeClassifier(criterion='entropy', class_weight='balanced', random_state=42)
    model.fit(X_tr, y_tr)
    print("ID3 Model trained successfully!")
    return model

def train_c45(X_tr, y_tr):
    """Trains a decision tree approximating C4.5 with balanced weights."""
    # Added class_weight='balanced' to handle class imbalance
    model = DecisionTreeClassifier(criterion='entropy', class_weight='balanced', random_state=42)
    model.fit(X_tr, y_tr)
    print("C4.5 Model trained successfully!")
    return model

def train_cart(X_tr, y_tr):
    """Trains a decision tree using the Gini Index with balanced weights."""
    # Added class_weight='balanced' to handle class imbalance
    model = DecisionTreeClassifier(criterion='gini', class_weight='balanced', random_state=42)
    model.fit(X_tr, y_tr)
    print("CART Model trained successfully!")
    return model

# Execute the functions using your imported data
if __name__ == "__main__":
    id3_model = train_id3(X_train, y_train)
    c45_model = train_c45(X_train, y_train)
    cart_model = train_cart(X_train, y_train)
