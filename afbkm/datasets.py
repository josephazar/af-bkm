"""Dataset loading + common preprocessing for NSL-KDD and UNSW-NB15.

Design choices (documented for the paper):
  * Binary task: 0 = benign/normal, 1 = attack/anomaly.
  * Continuous/numeric features only (categoricals dropped) so the Mahalanobis
    covariance stays well-conditioned and the model stays MCU-lightweight, matching
    the original Baseline K-means design.
  * Zero-/near-constant-variance columns are dropped (e.g. NSL-KDD num_outbound_cmds).
  * No preprocessing leakage: MinMax scaling to [0,1] is fit on the benign baseline
    ONLY and applied to the stream (see load(scale=...) and federated.run_simulation).
    A scale="global" (full-pool) option is kept solely for an appendix comparison.
  * Train/Test partitions of each dataset are pooled to form the federated stream;
    experiments do their own seeded shuffling and federated/sliding-window splits.
Processed arrays are cached to data/processed/<name>.npz (+ <name>.features.json).
"""
from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw")
PROC = os.path.join(ROOT, "data", "processed")
os.makedirs(PROC, exist_ok=True)

NSL_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty",
]
NSL_CATEGORICAL = ["protocol_type", "service", "flag"]
UNSW_DROP = ["id", "proto", "service", "state", "attack_cat", "label"]


def _select(X: np.ndarray, y: np.ndarray, names: list[str]):
    """Drop near-constant columns; return UNSCALED float32 features.

    Scaling is deliberately NOT applied here: to avoid train/test leakage it is
    fit on the benign baseline only, later (load(scale=...) or run_simulation).
    Variance-based column pruning is unsupervised and does not use labels.
    """
    X = np.asarray(X, dtype=np.float64)
    X[~np.isfinite(X)] = 0.0
    keep = X.std(axis=0) > 1e-8
    X = X[:, keep]
    names = [n for n, k in zip(names, keep) if k]
    return X.astype(np.float32), np.asarray(y, dtype=np.int64), names


def load_nsl_kdd():
    d = os.path.join(RAW, "nsl-kdd")
    frames = []
    for fn in ("KDDTrain+.txt", "KDDTest+.txt"):
        p = os.path.join(d, fn)
        if os.path.exists(p):
            frames.append(pd.read_csv(p, header=None, names=NSL_COLUMNS))
    if not frames:
        raise FileNotFoundError(f"NSL-KDD raw files not found in {d}")
    df = pd.concat(frames, ignore_index=True)
    y = (df["label"].astype(str) != "normal").astype(int).values
    feats = [c for c in NSL_COLUMNS if c not in NSL_CATEGORICAL + ["label", "difficulty"]]
    X = df[feats].apply(pd.to_numeric, errors="coerce").values
    return _select(X, y, feats)


def load_unsw_nb15():
    d = os.path.join(RAW, "unsw-nb15")
    frames = []
    for fn in ("UNSW_NB15_partA.csv", "UNSW_NB15_partB.csv"):
        p = os.path.join(d, fn)
        if os.path.exists(p):
            frames.append(pd.read_csv(p, low_memory=False))
    if not frames:
        raise FileNotFoundError(f"UNSW-NB15 raw files not found in {d}")
    df = pd.concat(frames, ignore_index=True)
    y = df["label"].astype(int).values
    feats = [c for c in df.columns if c not in UNSW_DROP]
    X = df[feats].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
    return _select(X, y, feats)


_LOADERS = {"nsl-kdd": load_nsl_kdd, "nsl_kdd": load_nsl_kdd,
            "unsw-nb15": load_unsw_nb15, "unsw_nb15": load_unsw_nb15}


def load(name: str, scale: str = "none", cache: bool = True):
    """Return (X float32 [N,d], y int {0,1} [N], feature_names list[str]).

    scale="none"   -> UNSCALED features; scaling is fit on the benign baseline
                      downstream (no leakage). This is the default used everywhere.
    scale="global" -> MinMax fit on the full pool (the leakage-prone protocol kept
                      only for the appendix comparison).
    """
    key = name.lower()
    if key not in _LOADERS:
        raise ValueError(f"unknown dataset {name!r}; options={sorted(_LOADERS)}")
    if scale not in ("none", "global"):
        raise ValueError(f"scale must be 'none' or 'global', got {scale!r}")
    canon = "nsl-kdd" if "nsl" in key else "unsw-nb15"
    npz = os.path.join(PROC, f"{canon}_{scale}.npz")
    fjson = os.path.join(PROC, f"{canon}.features.json")
    if cache and os.path.exists(npz) and os.path.exists(fjson):
        z = np.load(npz)
        with open(fjson) as fh:
            names = json.load(fh)
        return z["X"], z["y"], names
    X, y, names = _LOADERS[key]()
    if scale == "global":
        X = MinMaxScaler().fit_transform(X).astype(np.float32)
    np.savez_compressed(npz, X=X, y=y)
    with open(fjson, "w") as fh:
        json.dump(names, fh)
    return X, y, names


def summary(name: str) -> dict:
    X, y, names = load(name)
    return {
        "dataset": name, "n": int(X.shape[0]), "d": int(X.shape[1]),
        "n_features": len(names), "benign": int((y == 0).sum()),
        "attack": int((y == 1).sum()), "attack_ratio": float((y == 1).mean()),
    }


if __name__ == "__main__":
    for nm in ("nsl-kdd", "unsw-nb15"):
        print(summary(nm))
