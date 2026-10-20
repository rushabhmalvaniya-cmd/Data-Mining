"""
preprocess_02.py  -  ONLY preprocessing (feature scaling).

DBSCAN uses a distance threshold (eps), so features must be on the same scale.
"""
from sklearn.preprocessing import StandardScaler

from load_01 import load_all


def scale_all(datasets):
    """{name: (X, y)} -> {name: (X_scaled, y)}"""
    out = {}
    for name, (X, y) in datasets.items():
        out[name] = (StandardScaler().fit_transform(X), y)
    return out


if __name__ == "__main__":
    scaled = scale_all(load_all())
    for name, (X, _) in scaled.items():
        print(f"{name:6s} mean={X.mean(axis=0).round(3)}  std={X.std(axis=0).round(3)}")
