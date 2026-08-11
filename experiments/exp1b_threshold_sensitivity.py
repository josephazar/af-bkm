"""Experiment 1b - threshold sensitivity: motivates E1 (adaptive thresholding).

Shows that the standalone detector's F1/FPR swing strongly with the *manually*
chosen percentile, whereas search-free adaptive rules (MAD/Otsu/calibrated-FPR)
pick an operating point without a per-dataset search. They land close to the best
manual percentile on NSL-KDD and N-BaIoT; on UNSW-NB15 they stay well below it
(0.557 vs 0.726 F1), so "search-free" is not "optimal". Static single-node setting.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from afbkm import datasets, core, thresholds as T, metrics as M
import common as C

DATASETS = ["nsl-kdd", "unsw-nb15", "n-baiot"]
SEEDS = [0, 1, 2, 3, 4]
PCTS = [70, 75, 80, 85, 90, 95, 98]
ADAPT = [("MAD", "mad", {"k": 3.0}), ("Otsu", "otsu", {}),
         ("Calib@5%", "calibrated_quantile", {"target_fpr": 0.05})]
ACOL = {"MAD": "#2ca02c", "Otsu": "#9467bd", "Calib@5%": "#1f77b4"}


def run():
    curves, table = {}, []
    for ds in DATASETS:
        X, y, _ = datasets.load(ds)
        f1 = {p: [] for p in PCTS}
        fpr = {p: [] for p in PCTS}
        ad = {a[0]: {"f1": [], "fpr": []} for a in ADAPT}
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            perm = rng.permutation(len(X)); Xs, ys = X[perm], y[perm]
            bidx = np.where(ys == 0)[0][:2000]
            scaler = MinMaxScaler().fit(Xs[bidx])          # baseline-only (no leakage)
            Xb = scaler.transform(Xs[bidx])
            m, d = core.fit_baseline(Xb)
            dist = core.mahalanobis(scaler.transform(Xs), m.mean, m.inv_cov)
            for p in PCTS:
                pred = (dist > np.percentile(d, p)).astype(int)
                mt = M.binary_metrics(ys, pred)
                f1[p].append(mt["f1"]); fpr[p].append(mt["fpr"])
            for name, s, kw in ADAPT:
                pred = (dist > T.compute_threshold(d, strategy=s, **kw)).astype(int)
                mt = M.binary_metrics(ys, pred)
                ad[name]["f1"].append(mt["f1"]); ad[name]["fpr"].append(mt["fpr"])
        curves[ds] = (f1, fpr, ad)
        best_p = max(PCTS, key=lambda p: np.mean(f1[p]))
        row = {"Dataset": C.DATASET_TITLE[ds],
               "Best fixed-p F1": f"{np.mean(f1[best_p]):.3f} (p={best_p})",
               "F1 range over p": f"{min(np.mean(f1[p]) for p in PCTS):.3f}--{max(np.mean(f1[p]) for p in PCTS):.3f}"}
        for name, _, _ in ADAPT:
            row[f"{name} F1"] = f"{np.mean(ad[name]['f1']):.3f}"
        table.append(row)

    fig, axes = C.plt.subplots(1, len(DATASETS), figsize=(6 * len(DATASETS), 4.6))
    for ax, ds in zip(axes, DATASETS):
        f1, fpr, ad = curves[ds]
        mean = [np.mean(f1[p]) for p in PCTS]; std = [np.std(f1[p]) for p in PCTS]
        ax.errorbar(PCTS, mean, yerr=std, marker="o", color="#444", lw=2,
                    capsize=3, label="fixed percentile (manual)")
        for name, _, _ in ADAPT:
            ax.axhline(np.mean(ad[name]["f1"]), ls="--", lw=2, color=ACOL[name],
                       alpha=0.9, label=f"{name} (search-free)")
        ax.set_xlabel("manually chosen percentile $p$")
        ax.set_ylabel("F1-score")
        ax.set_title(C.DATASET_TITLE[ds])
        ax.legend(fontsize=9, loc="best")
    fig.suptitle("E1 motivation: F1 sensitivity to the manual threshold percentile "
                 "vs. search-free adaptive rules (mean$\\pm$std, 5 seeds)", y=1.03)
    C.savefig(fig, "exp1b_threshold_sensitivity")
    tab = pd.DataFrame(table)
    C.write_table(tab, "exp1b_threshold_sensitivity",
                  "Standalone-detector F1 under manual percentile selection vs. search-free "
                  "adaptive thresholds. The manual percentile must be searched per dataset; the "
                  "adaptive rules land near the best manual percentile on NSL-KDD and N-BaIoT, "
                  "whereas on UNSW-NB15 they provide a controlled operating point but do not "
                  "reach the best manual choice (0.557 vs. 0.726).", "tab:thr_sens")
    print(tab.to_string(index=False))


if __name__ == "__main__":
    run()
