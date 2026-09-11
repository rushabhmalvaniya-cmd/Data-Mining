from sklearn.naive_bayes import GaussianNB

# Import preprocessed features from Step 2
from step2_feature_encoding import X_train, y_train

# Initialize and train Gaussian Naive Bayes Model
model = GaussianNB()
model.fit(X_train, y_train)

print("\n--- Step 3: Model Training Completed ---")