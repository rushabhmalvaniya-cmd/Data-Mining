"""
load_01.py  -  ONLY loads data. No scaling, no models, no plots.

Datasets used to compare clustering quality metrics:
  * blobs : 4 Gaussian clusters with a little overlap (centroid-friendly)
  * moons : 2 interleaving half circles (density-friendly)
"""
from sklearn.datasets import make_blobs, make_moons

RANDOM_STATE = 42


def load_blobs(n_samples=600):
    X, y = make_blobs(n_samples=n_samples, centers=4, cluster_std=1.0,
                      random_state=RANDOM_STATE)
    return X, y


def load_moons(n_samples=600):
    X, y = make_moons(n_samples=n_samples, noise=0.07, random_state=RANDOM_STATE)
    return X, y


def load_all():
    """Return {name: (X, y_true)}"""
    return {"blobs": load_blobs(), "moons": load_moons()}


if __name__ == "__main__":
    for name, (X, y) in load_all().items():
        print(f"{name:6s} X={X.shape}  true clusters={len(set(y))}")
