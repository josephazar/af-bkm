"""Semi-supervised novelty-detection baselines (trained on benign data only).

These mirror the comparators in the published Baseline K-means evaluations and are
re-implemented with scikit-learn so no heavyweight DL dependency is required.
Each detector exposes fit(X_benign) / predict(X) -> {0 benign, 1 anomaly}.
"""
from __future__ import annotations
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, KernelDensity, NearestNeighbors


def _flip(pred) -> np.ndarray:
    """sklearn novelty convention (+1 inlier / -1 outlier) -> {0 benign, 1 anomaly}."""
    return (np.asarray(pred) == -1).astype(int)


class _SkNovelty:
    def __init__(self, name, model):
        self.name, self.model = name, model

    def fit(self, Xb):
        self.model.fit(np.asarray(Xb, dtype=float))
        return self

    def predict(self, X):
        return _flip(self.model.predict(np.asarray(X, dtype=float)))


class KDEDetector:
    def __init__(self, bandwidth=0.5, contamination=0.05):
        self.bandwidth, self.contamination = bandwidth, contamination

    def fit(self, Xb):
        Xb = np.asarray(Xb, dtype=float)
        self.kde = KernelDensity(bandwidth=self.bandwidth).fit(Xb)
        s = self.kde.score_samples(Xb)
        self.thr = np.percentile(s, 100 * self.contamination)  # low log-density => anomaly
        self.name = "KDE"
        return self

    def predict(self, X):
        return (self.kde.score_samples(np.asarray(X, dtype=float)) < self.thr).astype(int)


class KNNDetector:
    def __init__(self, k=5, contamination=0.05):
        self.k, self.contamination = k, contamination

    def fit(self, Xb):
        Xb = np.asarray(Xb, dtype=float)
        self.nn = NearestNeighbors(n_neighbors=self.k).fit(Xb)
        s = self.nn.kneighbors(Xb)[0].mean(axis=1)
        self.thr = np.percentile(s, 100 * (1 - self.contamination))  # far => anomaly
        self.name = "KNN"
        return self

    def predict(self, X):
        s = self.nn.kneighbors(np.asarray(X, dtype=float))[0].mean(axis=1)
        return (s > self.thr).astype(int)


def make_detectors(seed: int = 0, contamination: float = 0.05) -> dict:
    return {
        "OCSVM": _SkNovelty("OCSVM", OneClassSVM(kernel="rbf", nu=0.1, gamma="scale")),
        "IsolationForest": _SkNovelty(
            "IsolationForest",
            IsolationForest(n_estimators=150, contamination="auto", random_state=seed)),
        "LOF": _SkNovelty(
            "LOF", LocalOutlierFactor(n_neighbors=50, novelty=True, contamination="auto")),
        "KDE": KDEDetector(contamination=contamination),
        "KNN": KNNDetector(k=5, contamination=contamination),
    }
