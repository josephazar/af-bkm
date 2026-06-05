"""Experiment 5 - AF-BKM target-FPR operating-point sweep.

The benign-anchored threshold (E2) is calibrated to a target benign FPR, so the
operator can choose the precision/recall trade-off explicitly. This is especially
relevant on UNSW-NB15, where attacks are harder and recall is threshold-sensitive.
"""
import numpy as np
import pandas as pd
from afbkm import datasets, federated as F
import common as C

SEEDS = [0, 1, 2, 3, 4]
TARGETS = [0.05, 0.10, 0.15, 0.20]
SIM = dict(n_workers=3, n_baseline=2000, window=1000, merges=(6, 12, 18),
           alpha=0.5, max_epochs=24, blend=0.5)
MET = ["precision", "recall", "f1", "f2", "fpr"]


def _final(df):
    g = df.groupby("epoch")[MET].mean().tail(3).mean()
    return {m: float(g[m]) for m in MET}


def run():
    rows = []
    for ds in ["unsw-nb15", "nsl-kdd"]:
        X, y, _ = datasets.load(ds)
        for t in TARGETS:
            acc = {m: [] for m in MET}
            for s in SEEDS:
                df = F.run_simulation(
                    X, y, seed=s, aggregation="robust", consensus_mode="median",
                    trust_k=3.0, threshold_strategy="calibrated_quantile",
                    threshold_kwargs={"target_fpr": t}, **SIM)
                f = _final(df)
                for m in MET:
                    acc[m].append(f[m])
            rows.append(dict(dataset=ds, target_fpr=t,
                             **{m: np.mean(acc[m]) for m in MET},
                             **{m + "_std": np.std(acc[m]) for m in MET}))
        print(f"[exp5] {ds} done")
    R = pd.DataFrame(rows)
    R.to_csv(f"{C.TAB}/exp5_target_fpr.csv", index=False)

    # table
    tab = R.copy()
    for m in MET:
        tab[m] = tab.apply(lambda r: f"{r[m]:.3f} ± {r[m+'_std']:.3f}", axis=1)
    tab = tab[["dataset", "target_fpr"] + MET]
    tab["dataset"] = tab["dataset"].map(C.DATASET_TITLE)
    tab.columns = ["Dataset", "Target FPR", "Precision", "Recall", "F1", "F2", "Actual FPR"]
    C.write_table(tab, "exp5_target_fpr",
                  "AF-BKM operating points on the benign target-FPR knob (federated, mean$\\pm$std "
                  "over 5 seeds). The achieved FPR tracks the target, exposing an explicit "
                  "precision/recall trade-off, most pronounced on UNSW-NB15.", "tab:targetfpr")

    fig, axes = C.plt.subplots(1, 2, figsize=(13, 5))
    for ax, ds in zip(axes, ["unsw-nb15", "nsl-kdd"]):
        sub = R[R.dataset == ds].sort_values("target_fpr")
        for m, c in [("precision", "#1f77b4"), ("recall", "#2ca02c"), ("f1", "#ff7f0e")]:
            ax.errorbar(sub.target_fpr, sub[m], yerr=sub[m + "_std"], marker="o", lw=2,
                        capsize=3, color=c, label=m.capitalize())
        ax.set_xlabel("target benign FPR $\\phi$"); ax.set_ylabel("score")
        ax.set_title(C.DATASET_TITLE[ds]); ax.set_ylim(0, 1); ax.legend()
    fig.suptitle("AF-BKM precision/recall trade-off via the target-FPR knob", y=1.02)
    C.savefig(fig, "exp5_target_fpr")
    print(tab.to_string(index=False))


if __name__ == "__main__":
    run()
