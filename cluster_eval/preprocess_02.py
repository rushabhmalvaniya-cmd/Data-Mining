"""
preprocess_02.py  -  ONLY preprocessing (feature scaling).
"""
from sklearn.preprocessing import StandardScaler

from load_01 import load_all


def scale_all(datasets):
    """{name: (X, y)} -> {name: (X_scaled, y)}"""
    return {name: (StandardScaler().fit_transform(X), y)
            for name, (X, y) in datasets.items()}


if __name__ == "__main__":
    for name, (X, _) in scale_all(load_all()).items():
        print(f"{name:6s} mean={X.mean(axis=0).round(3)}  std={X.std(axis=0).round(3)}")
