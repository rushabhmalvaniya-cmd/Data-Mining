# step 5
from sklearn.neighbors import KNeighborsClassifier

def train_knn_euclidean(X_tr, y_tr, n_neighbors=5):
    """Trains a KNN Classifier on pixel features using Euclidean Distance."""
    model = KNeighborsClassifier(n_neighbors=n_neighbors, metric='euclidean')
    model.fit(X_tr, y_tr)
    print(f"KNN (k={n_neighbors}, Euclidean) Model trained successfully!")
    return model

def train_knn_manhattan(X_tr, y_tr, n_neighbors=5):
    """Trains a KNN Classifier on pixel features using Manhattan Distance."""
    model = KNeighborsClassifier(n_neighbors=n_neighbors, metric='manhattan')
    model.fit(X_tr, y_tr)
    print(f"KNN (k={n_neighbors}, Manhattan) Model trained successfully!")
    return model

if __name__ == "__main__":
    from train_test_split_knn_4 import X_train, y_train
    knn3_model = train_knn_euclidean(X_train, y_train, n_neighbors=3)
    knn5_model = train_knn_euclidean(X_train, y_train, n_neighbors=5)
    knn_manhattan_model = train_knn_manhattan(X_train, y_train, n_neighbors=5)