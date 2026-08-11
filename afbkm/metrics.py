"""Evaluation metrics for the AF-BKM experiments."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, fbeta_score,
    roc_auc_score, confusion_matrix,
)


def binary_metrics(y_true, y_pred, scores=None) -> dict:
    """Precision/Recall/F1/F2/FPR/Accuracy (+AUC if continuous scores given).

    Convention: 1 = anomaly/attack (positive class), 0 = benign.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    out = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
    }
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out["fpr"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    out["accuracy"] = float((tp + tn) / max(tp + tn + fp + fn, 1))
    out["support_pos"] = int(tp + fn)
    out["support_neg"] = int(tn + fp)
    if scores is not None and len(np.unique(y_true)) > 1:
        try:
            out["auc"] = float(roc_auc_score(y_true, scores))
        except Exception:
            out["auc"] = float("nan")
    return out


def precision_drop_per_merge(precision_series) -> float:
    """Mean per-step change in precision across merge rounds (negative = decay).

    Positive/near-zero indicates the merge no longer erodes precision (the E2 goal).
    """
    p = np.asarray(precision_series, dtype=float)
    if p.size < 2:
        return 0.0
    return float(np.mean(np.diff(p)))


def comms_floats_per_round(d: int, share_cov: bool = False) -> int:
    """Count the AF-BKM per-worker fp32 payload for one merge round.

    AF-BKM uploads a benign mean (d values), benign count, dispersion, and local
    threshold candidate (three scalars). ``share_cov`` adds the covariance upper
    triangle to this same payload for the communication-cost comparison.
    """
    if d <= 0:
        raise ValueError("d must be positive")
    base = d + 3
    if share_cov:
        base += d * (d + 1) // 2
    return int(base)
