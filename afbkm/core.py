"""Core AF-BKM detector: a vectorised Mahalanobis novelty model.

Clean re-implementation of the original `BaselineKMeans` Mahalanobis core
(baselineKmeans/utils/bkmeans.py) with three corrections:
  * fully vectorised distance (the original looped point-by-point in Python);
  * pluggable, regularised covariance inversion (ridge=original, diagonal/shrinkage
    as lightweight options) instead of a hard-coded pinv(cov+0.001*I);
  * pluggable threshold strategy (E1) instead of a fixed percentile.
Empty-input guards (the original produced NaN means for empty workers) live in
`federated.py`, which is where empty windows can occur.
"""
from __future__ import annotations
import numpy as np
from . import thresholds as T


def regularized_inverse(cov: np.ndarray, reg: float = 1e-3, method: str = "ridge") -> np.ndarray:
    cov = np.atleast_2d(np.asarray(cov, dtype=float))
    d = cov.shape[0]
    if method == "ridge":            # original idea (pinv(cov+reg*I)); reg makes it PD,
        A = cov + reg * np.eye(d)    # so a direct solve is exact and avoids pinv warnings
        try:
            return np.linalg.inv(A)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(A)
    if method == "diagonal":         # MCU-cheap: ignore off-diagonal covariance
        return np.diag(1.0 / (np.diag(cov) + reg))
    raise ValueError(f"unknown cov inverse method {method!r}")


def mahalanobis(X: np.ndarray, mean: np.ndarray, inv_cov: np.ndarray) -> np.ndarray:
    """Vectorised Mahalanobis distance of each row of X to `mean`."""
    diff = np.asarray(X, dtype=float) - np.asarray(mean, dtype=float)
    if diff.ndim == 1:
        diff = diff[None, :]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        m = np.einsum("ij,jk,ik->i", diff, inv_cov, diff)
    m = np.nan_to_num(m, nan=0.0, posinf=np.finfo(float).max, neginf=0.0)
    return np.sqrt(np.clip(m, 0.0, None))


class NoveltyModel:
    """Benign-baseline Mahalanobis model: mean, inverse covariance, threshold."""

    def __init__(self, mean, inv_cov, threshold, n=0, cov=None):
        self.mean = np.asarray(mean, dtype=float)
        self.inv_cov = np.asarray(inv_cov, dtype=float)
        self.threshold = float(threshold)
        self.n = int(n)
        self.cov = None if cov is None else np.asarray(cov, dtype=float)

    @property
    def d(self) -> int:
        return self.mean.shape[0]

    def distance(self, X) -> np.ndarray:
        return mahalanobis(X, self.mean, self.inv_cov)

    def predict(self, X) -> np.ndarray:
        return (self.distance(X) > self.threshold).astype(int)

    def feature_attribution(self, x) -> np.ndarray:
        """Per-feature contribution to the squared Mahalanobis distance (XAI teaser)."""
        diff = np.asarray(x, dtype=float) - self.mean
        return diff * (self.inv_cov @ diff)

    def copy(self) -> "NoveltyModel":
        return NoveltyModel(self.mean.copy(), self.inv_cov.copy(), self.threshold,
                            self.n, None if self.cov is None else self.cov.copy())


def fit_baseline(Xb, reg: float = 1e-3, cov_method: str = "ridge",
                 threshold_strategy: str = "percentile", threshold_kwargs=None):
    """Fit a NoveltyModel on benign baseline data; return (model, benign_distances)."""
    Xb = np.asarray(Xb, dtype=float)
    if Xb.ndim == 1:
        Xb = Xb[None, :]
    mean = Xb.mean(axis=0)
    if cov_method == "shrinkage":
        from sklearn.covariance import LedoitWolf
        cov = LedoitWolf().fit(Xb).covariance_
        inv = np.linalg.pinv(cov + reg * np.eye(cov.shape[0]))
    else:
        cov = np.cov(Xb.T)
        cov = np.atleast_2d(cov)
        inv = regularized_inverse(cov, reg=reg, method=cov_method)
    dist = mahalanobis(Xb, mean, inv)
    tau = T.compute_threshold(dist, strategy=threshold_strategy, **(threshold_kwargs or {}))
    return NoveltyModel(mean=mean, inv_cov=inv, threshold=tau, n=len(Xb), cov=cov), dist
