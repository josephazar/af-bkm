"""Experiment 3 - the Mahalanobis Baseline-K-means core vs. standard semi-supervised
novelty detectors (OCSVM/IsolationForest/LOF/KDE/KNN), trained on benign data and
evaluated on a held-out stream. Also records fit/predict time (lightweight claim).
Single-node (non-federated) comparison, mirroring the original papers.
"""
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from afbkm import datasets, core, baselines, metrics as M
import common as C

SEEDS = [0, 1, 2]
N_BASE = 2000
N_EVAL = 20000
DCOL = {"Bkmeans-Mahalanobis": "#2ca02c", "OCSVM": "#1f77b4", "IsolationForest": "#ff7f0e",
        "LOF": "#9467bd", "KDE": "#8c564b", "KNN": "#e377c2"}


def run():
    rows = []
    for ds in ["nsl-kdd", "unsw-nb15"]:
        X, y, _ = datasets.load(ds)
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            perm = rng.permutation(len(X)); Xs, ys = X[perm], y[perm]
            bidx = np.where(ys == 0)[0][:N_BASE]
            mask = np.ones(len(Xs), bool); mask[bidx] = False
            scaler = MinMaxScaler().fit(Xs[bidx])          # baseline-only (no leakage)
            Xb = scaler.transform(Xs[bidx])
            Xe, ye = scaler.transform(Xs[mask][:N_EVAL]), ys[mask][:N_EVAL]

            t0 = time.perf_counter()
            m, _ = core.fit_baseline(Xb, threshold_strategy="mad", threshold_kwargs={"k": 3.0})
            ft = time.perf_counter() - t0
            t0 = time.perf_counter(); pred = m.predict(Xe); pt = time.perf_counter() - t0
            rows.append(dict(dataset=ds, detector="Bkmeans-Mahalanobis", seed=seed,
                             fit_ms=1e3 * ft, pred_ms=1e3 * pt,
                             **M.binary_metrics(ye, pred, scores=m.distance(Xe))))
            for name, det in baselines.make_detectors(seed=seed).items():
                t0 = time.perf_counter(); det.fit(Xb); ft = time.perf_counter() - t0
                t0 = time.perf_counter(); pred = det.predict(Xe); pt = time.perf_counter() - t0
                rows.append(dict(dataset=ds, detector=name, seed=seed,
                                 fit_ms=1e3 * ft, pred_ms=1e3 * pt, **M.binary_metrics(ye, pred)))
        print(f"[exp3] {ds} done")
    R = pd.DataFrame(rows)
    R.to_csv(f"{C.TAB}/exp3_baselines_raw.csv", index=False)

    g = R.groupby(["dataset", "detector"]).agg(
        precision=("precision", "mean"), recall=("recall", "mean"),
        f1=("f1", "mean"), f2=("f2", "mean"), fpr=("fpr", "mean"),
        fit_ms=("fit_ms", "mean"), pred_ms=("pred_ms", "mean")).reset_index()

    # table
    tab = g.copy()
    for c in ["precision", "recall", "f1", "f2", "fpr"]:
        tab[c] = tab[c].map(lambda v: f"{v:.3f}")
    tab["fit_ms"] = tab["fit_ms"].map(lambda v: f"{v:.0f}")
    tab["pred_ms"] = tab["pred_ms"].map(lambda v: f"{v:.0f}")
    tab.columns = ["Dataset", "Detector", "Precision", "Recall", "F1", "F2", "FPR",
                   "Fit (ms)", "Predict (ms)"]
    tab["Dataset"] = tab["Dataset"].map(C.DATASET_TITLE)
    C.write_table(tab, "exp3_baselines",
                  "Standalone semi-supervised novelty detectors trained on 2000 benign samples "
                  "and evaluated on a 20k-packet stream (mean over 3 seeds). The Mahalanobis "
                  "Baseline-K-means core is competitive while being among the cheapest to fit/run.",
                  "tab:baselines")

    # bar chart: F1 per detector per dataset
    dets = ["Bkmeans-Mahalanobis", "OCSVM", "IsolationForest", "LOF", "KDE", "KNN"]
    short = {"Bkmeans-Mahalanobis": "Bkmeans\n(Mahal.)", "IsolationForest": "Isolation\nForest"}
    fig, axes = C.plt.subplots(1, 2, figsize=(15, 5.4))
    for ax, ds in zip(axes, ["nsl-kdd", "unsw-nb15"]):
        sub = g[g.dataset == ds].set_index("detector").reindex(dets)
        xpos = np.arange(len(dets)); w = 0.26
        for k, (metric, off) in enumerate([("precision", -w), ("recall", 0), ("f1", w)]):
            ax.bar(xpos + off, sub[metric].values, width=w, label=metric.capitalize())
        ax.set_xticks(xpos)
        ax.set_xticklabels([short.get(d, d) for d in dets], rotation=0, fontsize=10)
        ax.set_ylabel("score"); ax.set_title(C.DATASET_TITLE[ds]); ax.set_ylim(0, 1)
        ax.legend(fontsize=11, ncol=3, loc="upper center")
    fig.suptitle("Standalone novelty-detector comparison (trained on benign only)", y=1.02)
    C.savefig(fig, "exp3_baselines")
    print(tab.to_string(index=False))


if __name__ == "__main__":
    run()
