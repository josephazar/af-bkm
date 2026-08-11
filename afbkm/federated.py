"""Federated simulation harness + aggregation modes.

Two aggregation modes compare the published coordinator rule and the fix in the
same windowed simulation protocol:
  * mode="original": protocol-level reimplementation of the Sensors coordinator
    equations (equal-mean averaging + a percentile of benign distances filtered
    below the minimum worker-anomalous distance). Repeated min-outline filtering
    produces the precision-decay failure measured by the experiments.
  * mode="robust" (E2): count/quality-weighted worker mean blended with a trusted
    benign anchor; the threshold is RE-CALIBRATED on the trusted benign anchor (E1 rule),
    NOT taken as a worker consensus. Per-worker threshold candidates are used only for
    outlier-worker (trust-gate) filtering. Covariance is never transmitted (statistics-only).

Empty-worker guards fix the original NaN hazard (mean of an empty array).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from . import core, thresholds as T, metrics as M


def dirichlet_partition(strata, n_workers, alpha, rng):
    """Label-skew non-IID partition: split each stratum across workers by Dirichlet(alpha)."""
    strata = np.asarray(strata)
    worker_idx = [[] for _ in range(n_workers)]
    for c in np.unique(strata):
        idx = rng.permutation(np.where(strata == c)[0])
        props = rng.dirichlet([alpha] * n_workers)
        props = 0.85 * props + 0.15 / n_workers  # viability floor; Dirichlet skew stays dominant
        cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
        for w, part in enumerate(np.split(idx, cuts)):
            worker_idx[w].extend(part.tolist())
    return [rng.permutation(np.asarray(w, dtype=int)) for w in worker_idx]


def _worker_update(Xw, dist, model, threshold_strategy, threshold_kwargs,
                   poisoned=False, poison_kind="evasion"):
    """Label-free statistics a worker would upload (no ground-truth labels used)."""
    if poisoned:
        far = dist > model.threshold
        near = ~far
        bd = dist[near] if near.any() else dist
        if poison_kind == "evasion":          # claim all benign + inflate threshold
            bm, cand, bd, nb = Xw.mean(axis=0), float(np.max(dist)), dist, int(len(Xw))
        elif poison_kind == "mean_shift":     # report attack-region centroid as benign
            bm = Xw[far].mean(axis=0) if far.any() else Xw.mean(axis=0)
            cand = (float(T.compute_threshold(bd, strategy=threshold_strategy,
                                              **(threshold_kwargs or {}))) if bd.size > 1 else None)
            nb = int(near.sum())
        elif poison_kind == "thr_high":       # honest mean, grossly inflated candidate
            bm, cand, nb = (Xw[near].mean(axis=0) if near.any() else Xw.mean(axis=0)), float(np.max(dist) * 5), int(near.sum())
        elif poison_kind == "thr_low":        # honest mean, near-zero candidate
            bm, cand, nb = (Xw[near].mean(axis=0) if near.any() else Xw.mean(axis=0)), float(np.min(dist) * 0.1), int(near.sum())
        else:
            raise ValueError(f"unknown poison_kind {poison_kind!r}")
        return {"n_benign": max(nb, 1), "benign_mean": bm, "benign_distances": bd,
                "anom_min": np.inf,
                "dispersion": float(np.median(np.abs(bd - np.median(bd)))) if bd.size > 1 else np.nan,
                "threshold_candidate": cand}
    benign_mask = dist <= model.threshold
    anom_mask = ~benign_mask
    bdist = dist[benign_mask]
    n_benign = int(benign_mask.sum())
    upd = {
        "n_benign": n_benign,
        "benign_mean": Xw[benign_mask].mean(axis=0) if n_benign > 0 else None,
        "benign_distances": bdist if n_benign > 0 else np.array([]),
        "anom_min": float(dist[anom_mask].min()) if anom_mask.any() else np.inf,
        "dispersion": float(np.median(np.abs(bdist - np.median(bdist)))) if n_benign > 1 else np.nan,
        "threshold_candidate": (T.compute_threshold(bdist, strategy=threshold_strategy,
                                                     **(threshold_kwargs or {}))
                                if n_benign > 1 else None),
    }
    return upd


def _aggregate(model, coord, updates, *, mode, threshold_strategy, threshold_kwargs,
               consensus_mode, quality_weight, trust_k, blend):
    valid = [u for u in updates if u["benign_mean"] is not None and u["n_benign"] > 0]
    if not valid:
        return model  # nothing to merge (all-empty); leave model unchanged (NaN guard)

    if mode == "original":
        # Equal-mean coordinator rule evaluated on this simulation window.
        means = [coord["mean"]] + [u["benign_mean"] for u in valid]
        new_mean = np.mean(np.stack(means, axis=0), axis=0)
        inv = model.inv_cov  # coordinator covariance, never updated by workers
        coord_dist = core.mahalanobis(coord["Xb"], new_mean, inv)
        inline = np.concatenate([coord_dist] + [u["benign_distances"] for u in valid])
        min_outline = min(u["anom_min"] for u in valid)
        filt = inline[inline < min_outline] if np.isfinite(min_outline) else inline
        if filt.size == 0:
            filt = inline
        new_tau = T.compute_threshold(filt, strategy=threshold_strategy, **(threshold_kwargs or {}))
        coord["mean"] = new_mean
        return core.NoveltyModel(new_mean, inv, new_tau, n=model.n, cov=model.cov)

    if mode == "robust":
        # outlier-worker filtering on threshold candidates (trust gate)
        cands = np.array([u["threshold_candidate"] for u in valid
                          if u["threshold_candidate"] is not None], dtype=float)
        keep = valid
        if cands.size >= 3 and trust_k is not None:
            med = np.median(cands)
            mad = np.median(np.abs(cands - med)) * 1.4826 + 1e-12
            keep = [u for u in valid if u["threshold_candidate"] is not None
                    and abs(u["threshold_candidate"] - med) <= trust_k * mad]
            if not keep:
                keep = valid
        # count/quality-weighted mean of worker benign means
        w = np.array([float(u["n_benign"]) for u in keep], dtype=float)
        if quality_weight:
            disp = np.array([0.0 if not np.isfinite(u["dispersion"]) else u["dispersion"] for u in keep])
            w = w / (1.0 + disp)
        w = w / w.sum()
        worker_mean = np.tensordot(w, np.stack([u["benign_mean"] for u in keep], axis=0), axes=(0, 0))
        # Blend the accepted worker mean with the fixed commissioning anchor.
        new_mean = (1.0 - blend) * coord["anchor_mean"] + blend * worker_mean
        inv = model.inv_cov  # statistics-only: covariance is never transmitted
        # benign-anchored threshold: recalibrate on the TRUSTED benign baseline to a
        # fixed target FPR (E1 rule), replacing min-outline tightening -> precision stays put
        base_dist = core.mahalanobis(coord["Xb"], new_mean, inv)
        new_tau = T.compute_threshold(base_dist, strategy=threshold_strategy, **(threshold_kwargs or {}))
        coord["mean"] = new_mean
        return core.NoveltyModel(new_mean, inv, float(new_tau), n=model.n, cov=model.cov)

    raise ValueError(f"unknown aggregation mode {mode!r}")


def run_simulation(X, y, *, n_workers=3, n_baseline=2000, window=1000,
                   merges=(7, 14, 21), alpha=1.0, seed=0,
                   reg=1e-3, cov_method="ridge",
                   threshold_strategy="percentile", threshold_kwargs=None,
                   aggregation="original", consensus_mode="median",
                   quality_weight=True, trust_k=3.0, blend=0.5, strata=None,
                   poison_workers=(), poison_kind="evasion", scale_on_baseline=True,
                   contaminate_baseline=0.0, max_epochs=None, label=""):
    """Run one federated streaming simulation; return a per-epoch per-worker DataFrame."""
    if not 0.0 <= contaminate_baseline <= 1.0:
        raise ValueError("contaminate_baseline must be between 0 and 1")
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=int)
    perm = rng.permutation(len(X)); X, y = X[perm], y[perm]
    s = (y if strata is None else np.asarray(strata)[perm])

    benign_idx = np.where(y == 0)[0]
    if contaminate_baseline > 0.0:
        # Replace a fraction of the commissioning anchor with seeded attack rows.
        # The p=0 branch remains bit-identical to the uncontaminated protocol.
        attack_idx = np.where(y == 1)[0]
        n_attack = int(round(contaminate_baseline * n_baseline))
        n_benign = n_baseline - n_attack
        if n_benign > len(benign_idx) or n_attack > len(attack_idx):
            raise ValueError("not enough benign/attack rows for the requested contaminated anchor")
        base_sel = np.concatenate([benign_idx[:n_benign], attack_idx[:n_attack]])
    else:
        if n_baseline > len(benign_idx):
            raise ValueError("not enough benign rows for the requested baseline")
        base_sel = benign_idx[:n_baseline]
    Xb = X[base_sel]
    mask = np.ones(len(X), dtype=bool); mask[base_sel] = False
    Xs, ys, ss = X[mask], y[mask], s[mask]

    # No-leakage scaling: fit the scaler ONLY on the coordinator's benign baseline,
    # then transform the (future) stream with it.
    if scale_on_baseline:
        scaler = MinMaxScaler().fit(Xb)
        Xb = scaler.transform(Xb).astype(np.float32)
        Xs = scaler.transform(Xs).astype(np.float32)

    parts = dirichlet_partition(ss, n_workers, alpha, rng)
    model, base_dist = core.fit_baseline(Xb, reg=reg, cov_method=cov_method,
                                          threshold_strategy=threshold_strategy,
                                          threshold_kwargs=threshold_kwargs)
    coord = {"mean": model.mean.copy(), "anchor_mean": model.mean.copy(),
             "Xb": Xb, "n": len(Xb)}

    n_epochs = min(len(p) for p in parts) // window
    if max_epochs:
        n_epochs = min(n_epochs, max_epochs)
    ptr = [0] * n_workers
    rows = []
    tkw = threshold_kwargs or {}
    for epoch in range(1, n_epochs + 1):
        updates = []
        merged_now = epoch in merges
        for w in range(n_workers):
            sl = parts[w][ptr[w]:ptr[w] + window]; ptr[w] += window
            Xw, yw = Xs[sl], ys[sl]
            dist = model.distance(Xw)
            pred = (dist > model.threshold).astype(int)
            mt = M.binary_metrics(yw, pred, scores=dist)
            mt.update(dict(epoch=epoch, worker=f"W{w}", threshold=float(model.threshold),
                           is_merge_epoch=merged_now, aggregation=aggregation,
                           threshold_strategy=threshold_strategy, alpha=alpha,
                           n_workers=n_workers, seed=seed, label=label))
            rows.append(mt)
            updates.append(_worker_update(Xw, dist, model, threshold_strategy, tkw,
                                          poisoned=(w in poison_workers),
                                          poison_kind=poison_kind))
        if merged_now:
            model = _aggregate(model, coord, updates, mode=aggregation,
                               threshold_strategy=threshold_strategy, threshold_kwargs=tkw,
                               consensus_mode=consensus_mode, quality_weight=quality_weight,
                               trust_k=trust_k, blend=blend)
    return pd.DataFrame(rows)
