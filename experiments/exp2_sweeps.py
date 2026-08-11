"""Experiment 2 - robustness sweeps over non-IID severity (Dirichlet alpha) and
number of workers, comparing Baseline (original merge) vs AF-BKM (E1+E2).
Reports final precision (last 3 epochs) and mean FPR.
"""
import numpy as np
import pandas as pd
from afbkm import datasets, federated as F
import common as C

SEEDS = [0, 1, 2, 3, 4]
CFG = {
    "Baseline": dict(aggregation="original", threshold_strategy="percentile",
                     threshold_kwargs={"percentile": 90}),
    "AF-BKM": dict(aggregation="robust", consensus_mode="median",
                   threshold_strategy="mad", threshold_kwargs={"k": 3.0}),
}
SIM = dict(n_baseline=2000, window=300, merges=(2, 4, 6), max_epochs=8, blend=0.5)
ALPHAS = [0.1, 0.3, 0.5, 1.0, 5.0]
WORKERS = [3, 5, 10]
MCOL = {"Baseline": "#d62728", "AF-BKM": "#2ca02c"}


def _final_prec(df):
    return df.groupby("epoch").precision.mean().tail(3).mean()


def run():
    rows = []
    for ds in ["nsl-kdd", "unsw-nb15", "n-baiot"]:
        X, y, _ = datasets.load(ds)
        for a in ALPHAS:
            for name, cfg in CFG.items():
                fps, fprs = [], []
                for s in SEEDS:
                    df = F.run_simulation(X, y, n_workers=3, alpha=a, seed=s, **cfg, **SIM)
                    fps.append(_final_prec(df)); fprs.append(df.fpr.mean())
                rows.append(dict(dataset=ds, sweep="alpha", x=a, method=name,
                                 prec=np.mean(fps), prec_std=np.std(fps), fpr=np.mean(fprs)))
        for nw in WORKERS:
            for name, cfg in CFG.items():
                fps, fprs = [], []
                for s in SEEDS:
                    df = F.run_simulation(X, y, n_workers=nw, alpha=0.5, seed=s, **cfg, **SIM)
                    fps.append(_final_prec(df)); fprs.append(df.fpr.mean())
                rows.append(dict(dataset=ds, sweep="workers", x=nw, method=name,
                                 prec=np.mean(fps), prec_std=np.std(fps), fpr=np.mean(fprs)))
        print(f"[exp2] {ds} done")
    R = pd.DataFrame(rows)
    R.to_csv(f"{C.TAB}/exp2_sweeps.csv", index=False)

    fig, axes = C.plt.subplots(3, 2, figsize=(12, 12.5))
    for i, ds in enumerate(["nsl-kdd", "unsw-nb15", "n-baiot"]):
        for j, (sweep, xlab) in enumerate([("alpha", "Dirichlet $\\alpha$ (smaller = more non-IID)"),
                                           ("workers", "number of workers")]):
            ax = axes[i, j]
            sub = R[(R.dataset == ds) & (R.sweep == sweep)]
            for name in CFG:
                s = sub[sub.method == name].sort_values("x")
                ax.errorbar(s.x, s.prec, yerr=s.prec_std, marker="o", lw=2, capsize=3,
                            color=MCOL[name], label=name)
            if sweep == "alpha":
                ax.set_xscale("log")
            ax.set_xlabel(xlab); ax.set_ylabel("final precision (last 3 epochs)")
            ax.set_title(f"{C.DATASET_TITLE[ds]} - vs {sweep}")
            ax.legend()
    fig.suptitle("AF-BKM keeps precision high and stable across non-IID severity and worker counts",
                 y=1.01)
    C.savefig(fig, "exp2_sweeps")
    print(R.round(3).to_string(index=False))


if __name__ == "__main__":
    run()
