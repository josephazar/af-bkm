"""Dataset loading and common preprocessing for the AF-BKM experiments.

Design choices used by the experiments:
  * Binary task: 0 = benign/normal, 1 = attack/anomaly.
  * Continuous/numeric features only (categoricals dropped) so the Mahalanobis
    covariance stays well-conditioned and the model remains compact, matching the
    original Baseline K-means design.
  * Zero-/near-constant-variance columns are dropped (e.g. NSL-KDD num_outbound_cmds).
  * No preprocessing leakage: MinMax scaling to [0,1] is fit on the benign baseline
    ONLY and applied to the stream (see load(scale=...) and federated.run_simulation).
    A scale="global" (full-pool) option is kept solely for an appendix comparison.
    One disclosed exception: the near-constant column filter in _select() runs
    over the pooled corpus before experiment splits. It is label-blind full-corpus
    preprocessing. A recompute on all benign rows gives the same retained-column
    mask on the three corpora, but this is not a strictly inductive baseline-only
    step.
  * Train/Test partitions of each dataset are pooled to form the federated stream;
    experiments do their own seeded shuffling and federated/sliding-window splits.
Processed arrays are cached to data/processed/<name>.npz (+ <name>.features.json).
"""
from __future__ import annotations
import hashlib
import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_ROOT = os.path.abspath(os.environ.get("AFBKM_DATA_DIR", os.path.join(ROOT, "data")))
RAW = os.path.join(DATA_ROOT, "raw")
PROC = os.path.join(DATA_ROOT, "processed")
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
EXPECTED_CORPORA = {
    "nsl-kdd": (148517, 37, 77054, 71463),
    "unsw-nb15": (257673, 39, 93000, 164673),
    "n-baiot": (136565, 115, 71716, 64849),
}

# Bump when the pooling order or feature selection of any corpus changes; caches
# written by an older version are ignored rather than silently reused.
CACHE_VERSION = "v2"

# sha256 over (X float32 bytes || y int64 bytes) of the published corpora. Pins
# row ORDER as well as content, so a reordered traversal fails loudly instead of
# producing different-but-plausible numbers. Printed by `python -m afbkm.datasets`.
_CORPUS_SHA256 = {
    "nsl-kdd": "36d9ede510f63bbb127ce1e2b33022e633be88a6c891373e921a1d1dadcb5a52",
    "unsw-nb15": "7777d0fff954bfc8ae6fa5996eeb174ce18d04f12a409f255be699344542cbc5",
    "n-baiot": "3d857d4a98ff2924eaa7bb48e2101ab386ba90d1739b694ef79c88807fe4e2bd",
}


def _select(X: np.ndarray, y: np.ndarray, names: list[str]):
    """Drop near-constant columns; return UNSCALED float32 features.

    Scaling is deliberately NOT applied here: to avoid train/test leakage it is
    fit on the benign baseline only, later (load(scale=...) or run_simulation).

    The pruning below is computed over whatever matrix it is handed, which is the
    pooled corpus in the public experiments. It reads X only and never y, so it is
    label-blind, but it is disclosed as full-corpus preprocessing rather than as a
    baseline-only step.
    """
    X = np.array(X, dtype=np.float64, copy=True)  # pandas 3 may return read-only views
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
    if len(frames) != 2:
        raise FileNotFoundError(f"Both NSL-KDD train and test files are required in {d}")
    df = pd.concat(frames, ignore_index=True)
    y = (df["label"].astype(str) != "normal").astype(int).values
    feats = [c for c in NSL_COLUMNS if c not in NSL_CATEGORICAL + ["label", "difficulty"]]
    X = df[feats].apply(pd.to_numeric, errors="coerce").values
    return _select(X, y, feats)


def load_unsw_nb15():
    d = os.path.join(RAW, "unsw-nb15")
    frames = []
    alternatives = (
        ("UNSW_NB15_training-set.csv", "UNSW_NB15_partA.csv"),
        ("UNSW_NB15_testing-set.csv", "UNSW_NB15_partB.csv"),
    )
    for choices in alternatives:
        path = next((os.path.join(d, name) for name in choices
                     if os.path.exists(os.path.join(d, name))), None)
        if path:
            frames.append(pd.read_csv(path, low_memory=False))
    if len(frames) != len(alternatives):
        raise FileNotFoundError(
            f"Both UNSW-NB15 training and testing CSVs are required in {d}; see README.md"
        )
    df = pd.concat(frames, ignore_index=True)
    y = df["label"].astype(int).values
    feats = [c for c in df.columns if c not in UNSW_DROP]
    X = df[feats].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
    return _select(X, y, feats)


def load_nbaiot():
    """Load the exact FedMSE non-IID 50-client N-BaIoT preparation.

    Each client contributes headerless 115-feature CSV files in ``normal``,
    ``test_normal``, and ``abnormal`` directories. The two normal partitions are
    benign (0), and the abnormal partition is attack (1). Client partitions are
    pooled here; experiments subsequently apply their own seeded partitioning.

    Row order is CANONICAL and must not be changed: files are read subset-major
    (all clients' ``normal``, then all ``test_normal``, then all ``abnormal``),
    each group in sorted client order. Pooling order fixes the row order of the
    corpus, which the seeded per-experiment permutation then consumes, so a
    different traversal (for example client-major) yields a different, equally
    valid but NON-REPRODUCING split. The published results use the ordering
    below; ``_CORPUS_SHA256`` pins it.
    """
    import glob

    root = os.path.join(RAW, "n-baiot")
    scenario = os.path.join(root, "nonIID-50")
    if not os.path.isdir(scenario):
        scenario = root

    clients = sorted(glob.glob(os.path.join(scenario, "Client-*")))
    if len(clients) != 50:
        raise FileNotFoundError(
            "Expected the 50 FedMSE client directories under "
            f"{scenario}, found {len(clients)}. See README.md for preparation steps."
        )

    frames, ys = [], []
    for subset, label in (("normal", 0), ("test_normal", 0), ("abnormal", 1)):
        for client in clients:
            path = os.path.join(client, subset, "data.csv")
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Missing FedMSE N-BaIoT partition: {path}")
            frame = pd.read_csv(path, header=None)
            if frame.shape[1] != 115:
                raise ValueError(f"Expected 115 N-BaIoT features in {path}, found {frame.shape[1]}")
            frames.append(frame)
            ys.append(np.full(len(frame), label, dtype=np.int64))

    X = pd.concat(frames, ignore_index=True).apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0.0).values
    y = np.concatenate(ys)
    names = [f"f{i + 1}" for i in range(X.shape[1])]
    return _select(X, y, names)


_LOADERS = {"nsl-kdd": load_nsl_kdd, "nsl_kdd": load_nsl_kdd,
            "unsw-nb15": load_unsw_nb15, "unsw_nb15": load_unsw_nb15,
            "n-baiot": load_nbaiot, "n_baiot": load_nbaiot, "nbaiot": load_nbaiot}
_CANON = {"nsl-kdd": "nsl-kdd", "nsl_kdd": "nsl-kdd",
          "unsw-nb15": "unsw-nb15", "unsw_nb15": "unsw-nb15",
          "n-baiot": "n-baiot", "n_baiot": "n-baiot", "nbaiot": "n-baiot"}


def corpus_digest(X, y) -> str:
    """Content fingerprint of a loaded corpus, sensitive to row ORDER.

    Shape and class counts alone cannot detect a reordered pooling traversal,
    which silently changes every seeded split downstream. This digest can.
    """
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(X, dtype=np.float32).tobytes())
    h.update(np.ascontiguousarray(y, dtype=np.int64).tobytes())
    return h.hexdigest()


def _validate_corpus(canon, X, y, names, check_digest: bool = True):
    expected = EXPECTED_CORPORA[canon]
    observed = (len(X), X.shape[1], int((y == 0).sum()), int((y == 1).sum()))
    if observed != expected or len(names) != X.shape[1]:
        raise ValueError(
            f"{canon} does not match the published corpus; expected "
            f"(rows, features, benign, attack)={expected}, found {observed}."
        )
    want = _CORPUS_SHA256.get(canon) if check_digest else None
    if want:
        got = corpus_digest(X, y)
        if got != want:
            raise ValueError(
                f"{canon} content/order digest mismatch: expected {want}, got {got}. "
                "The corpus contents or the pooling order differ from the published "
                "run; delete data/processed/ and re-stage the raw inputs per README.md."
            )


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
    canon = _CANON[key]
    npz = os.path.join(PROC, f"{canon}_{scale}_{CACHE_VERSION}.npz")
    fjson = os.path.join(PROC, f"{canon}_{CACHE_VERSION}.features.json")
    if cache and os.path.exists(npz) and os.path.exists(fjson):
        with np.load(npz) as z:
            X, y = z["X"], z["y"]
            stamped_raw = str(z["raw_sha256"]) if "raw_sha256" in z else None
            stamped_self = str(z["data_sha256"]) if "data_sha256" in z else None
        with open(fjson) as fh:
            names = json.load(fh)
        if scale == "none":
            # An unscaled cache IS the raw corpus, so hash it against the pin.
            _validate_corpus(canon, X, y, names)
        else:
            # A transformed cache cannot be hashed against the raw pin, so it
            # carries two stamps: the digest of the raw corpus it was derived from,
            # and the digest of its own stored contents. Both are verified, so a
            # scaled cache is never accepted on structure alone. Caches written
            # before these fields existed are rejected rather than trusted.
            _validate_corpus(canon, X, y, names, check_digest=False)
            want = _CORPUS_SHA256.get(canon)
            if want and stamped_raw != want:
                raise ValueError(
                    f"{canon} scale={scale!r} cache carries raw digest {stamped_raw!r}, "
                    f"expected {want!r}; delete data/processed/ and rebuild."
                )
            if stamped_self is None or stamped_self != corpus_digest(X, y):
                raise ValueError(
                    f"{canon} scale={scale!r} cache contents do not match their own "
                    "digest; delete data/processed/ and rebuild."
                )
        return X, y, names
    X, y, names = _LOADERS[key]()
    _validate_corpus(canon, X, y, names)  # always digest-check the raw source
    raw_sha = corpus_digest(X, y)
    if scale == "global":
        X = MinMaxScaler().fit_transform(X).astype(np.float32)
    np.savez_compressed(npz, X=X, y=y, raw_sha256=np.array(raw_sha),
                        data_sha256=np.array(corpus_digest(X, y)))
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
    for nm in ("nsl-kdd", "unsw-nb15", "n-baiot"):
        try:
            X, y, _ = load(nm)
            print(summary(nm), "sha256=" + corpus_digest(X, y))
        except FileNotFoundError as exc:
            print(f"{nm}: SKIPPED ({exc})")
