"""E1 - Adaptive auto-thresholding for the benign Mahalanobis-distance distribution.

All strategies take a 1-D array of *benign* Mahalanobis distances and return a
scalar decision threshold tau. A new observation x is flagged anomalous iff
D_M(x) > tau.

`consensus()` aggregates per-worker scalar threshold candidates in the federated
setting (E1 federated threshold consensus), robust to outlier workers.
"""
from __future__ import annotations
import numpy as np


def percentile_threshold(d, percentile: float = 90.0) -> float:
    """Original Baseline K-means rule: fixed percentile of benign distances."""
    return float(np.percentile(d, percentile))


def mad_threshold(d, k: float = 3.0) -> float:
    """Robust median + k*scaled-MAD threshold (resistant to heavy tails)."""
    d = np.asarray(d, dtype=float)
    med = np.median(d)
    mad = np.median(np.abs(d - med))
    return float(med + k * 1.4826 * mad)


def calibrated_quantile_threshold(d, target_fpr: float = 0.05) -> float:
    """Pick the quantile that admits at most `target_fpr` benign false positives."""
    target_fpr = float(min(max(target_fpr, 1e-4), 0.5))
    return float(np.quantile(d, 1.0 - target_fpr))


def otsu_threshold(d, bins: int = 256) -> float:
    """Otsu between-class variance maximisation on the benign-distance histogram."""
    d = np.asarray(d, dtype=float)
    if d.size == 0:
        return 0.0
    hist, edges = np.histogram(d, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    total = hist.sum()
    if total == 0:
        return float(np.percentile(d, 90))
    w = np.cumsum(hist).astype(float)
    cumsum = np.cumsum(hist * centers)
    grand = cumsum[-1]
    best_t, best_var = float(centers[-1]), -1.0
    for i in range(len(centers)):
        wB = w[i] / total
        if wB <= 0.0 or wB >= 1.0:
            continue
        muB = cumsum[i] / w[i]
        muF = (grand - cumsum[i]) / (total - w[i])
        var = wB * (1.0 - wB) * (muB - muF) ** 2
        if var > best_var:
            best_var, best_t = var, float(centers[i])
    return best_t


def kneedle_threshold(d) -> float:
    """Knee of the sorted-distance curve (max vertical gap above the chord)."""
    s = np.sort(np.asarray(d, dtype=float))
    n = s.size
    if n < 3:
        return float(s[-1]) if n else 0.0
    x = np.linspace(0.0, 1.0, n)
    rng = s.max() - s.min()
    y = (s - s.min()) / (rng + 1e-12)
    # benign-distance CDF is convex (long upper tail): the knee is the point of
    # maximum vertical gap *below* the (0,0)-(1,1) chord, i.e. argmax(x - y).
    idx = int(np.argmax(x - y))
    return float(s[idx])


STRATEGIES = {
    "percentile": percentile_threshold,
    "mad": mad_threshold,
    "calibrated_quantile": calibrated_quantile_threshold,
    "otsu": otsu_threshold,
    "kneedle": kneedle_threshold,
}


def compute_threshold(d, strategy: str = "percentile", **kwargs) -> float:
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown threshold strategy {strategy!r}; options={list(STRATEGIES)}")
    return STRATEGIES[strategy](d, **kwargs)


def consensus(values, mode: str = "median", trim: float = 0.1):
    """Federated consensus over per-worker scalar threshold candidates.

    mode='min' replicates the original coordinator behaviour (over-conservative);
    'median'/'trimmed'/'mean' are the robust E1+E2 alternatives.
    """
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return None
    if mode == "median":
        return float(np.median(v))
    if mode == "mean":
        return float(np.mean(v))
    if mode == "min":
        return float(np.min(v))
    if mode == "trimmed":
        v = np.sort(v)
        k = int(v.size * trim)
        core = v[k: v.size - k] if v.size - 2 * k > 0 else v
        return float(np.mean(core))
    raise ValueError(f"unknown consensus mode {mode!r}")
