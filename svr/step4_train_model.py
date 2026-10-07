from sklearn.svm import SVR

def train_svr(X_train_scaled, y_train_scaled, C=10.0, epsilon=0.2):
    model = SVR(kernel='linear', C=C, epsilon=epsilon)
    model.fit(X_train_scaled, y_train_scaled)
    return model