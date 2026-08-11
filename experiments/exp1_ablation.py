"""Experiment 1 - ablation {Baseline, E1, E2, E1+E2} across three datasets and seeds.

Produces the headline precision/F1/FPR-vs-merge figures and the ablation table,
demonstrating that E2 (robust benign-anchored aggregation) removes the
merge-induced precision decay.
"""
import numpy as np
import pandas as pd
from afbkm import datasets, federated as F
import common as C

CONFIGS = {
    "Baseline (original)": dict(aggregation="original", threshold_strategy="percentile",
                                threshold_kwargs={"percentile": 90}),
    "E1 adaptive-thr": dict(aggregation="original", threshold_strategy="mad",
                            threshold_kwargs={"k": 3.0}),
    "E2 robust-agg": dict(aggregation="robust", consensus_mode="median",
                          threshold_strategy="percentile", threshold_kwargs={"percentile": 90}),
    "AF-BKM (E1+E2)": dict(aggregation="robust", consensus_mode="median",
                           threshold_strategy="mad", threshold_kwargs={"k": 3.0}),
}
DATASETS = ["nsl-kdd", "unsw-nb15", "n-baiot"]
SEEDS = list(range(10))
SIM = dict(n_workers=3, n_baseline=2000, window=1000, merges=(6, 12, 18),
           alpha=0.5, max_epochs=24, blend=0.5, quality_weight=True, trust_k=3.0)


def run():
    rows = []
    for ds in DATASETS:
        X, y, _ = datasets.load(ds)
        for cfg_name, cfg in CONFIGS.items():
            for seed in SEEDS:
                df = F.run_simulation(X, y, seed=seed, label=cfg_name, **cfg, **SIM)
                df["dataset"] = ds
                df["config"] = cfg_name
                rows.append(df)
            print(f"[exp1] {ds:10s} | {cfg_name:22s} done ({len(SEEDS)} seeds)")
    raw = pd.concat(rows, ignore_index=True)
    raw.to_csv(f"{C.TAB}/exp1_ablation_raw.csv", index=False)

    pe = C.per_epoch(raw)
    ms = C.mean_std_over_seeds(pe)

    # ---- figures: precision, F1, FPR vs epoch, one row of 3 panels per dataset ----
    order = list(CONFIGS.keys())
    for ds in DATASETS:
        fig, axes = C.plt.subplots(1, 3, figsize=(15, 4.2))
        for ax, metric, ylab in zip(axes, ["precision", "f1", "fpr"],
                                    ["Precision", "F1-score", "False-Positive Rate"]):
            sub = ms[ms.dataset == ds]
            for cfg in order:
                s = sub[sub.config == cfg].sort_values("epoch")
                ax.plot(s.epoch, s[f"{metric}_mean"], label=cfg, color=C.PALETTE[cfg], lw=2)
                ax.fill_between(s.epoch, s[f"{metric}_mean"] - s[f"{metric}_std"],
                                s[f"{metric}_mean"] + s[f"{metric}_std"],
                                color=C.PALETTE[cfg], alpha=0.12)
            for m in SIM["merges"]:
                ax.axvline(m, color="grey", ls="--", lw=1, alpha=0.6)
            ax.set_xlabel("Epoch (window of 1000 packets)")
            ax.set_ylabel(ylab)
            ax.set_title(ylab)
        axes[0].legend(loc="lower left", framealpha=0.9)
        fig.suptitle(f"{C.DATASET_TITLE[ds]}: effect of merge operations "
                     f"(dashed = merge; mean$\\pm$std over {len(SEEDS)} seeds)", y=1.02)
        C.savefig(fig, f"exp1_metrics_vs_merge_{ds}")

    # ---- ablation summary table ----
    summary = []
    for ds in DATASETS:
        for cfg in order:
            sub = pe[(pe.dataset == ds) & (pe.config == cfg)]
            recs = {"Dataset": C.DATASET_TITLE[ds], "Config": cfg}
            # early vs late precision per seed -> delta (decay if negative)
            deltas, finals = [], []
            for seed in SEEDS:
                s = sub[sub.seed == seed].sort_values("epoch")
                early = s.precision.head(3).mean()
                late = s.precision.tail(3).mean()
                deltas.append(late - early)
                finals.append(late)
            recs["Precision (final)"] = f"{np.mean(finals):.3f} {C.PM} {np.std(finals):.3f}"
            recs[f"{C.DELTA}P (early{C.ARROW}late)"] = f"{np.mean(deltas):+.3f}"
            for m, lab in [("recall", "Recall"), ("f1", "F1"), ("f2", "F2"), ("fpr", "FPR")]:
                recs[lab] = f"{sub[m].mean():.3f} {C.PM} {sub.groupby('seed')[m].mean().std():.3f}"
            summary.append(recs)
    tab = pd.DataFrame(summary)
    C.write_table(tab, "exp1_ablation_summary",
                  f"Ablation over the two mechanisms on NSL-KDD, UNSW-NB15 and N-BaIoT "
                  f"(3 workers, 3 merges, mean$\\pm$std over {len(SEEDS)} seeds). $\\Delta$P is the change in "
                  "precision from the first to the last three epochs; negative indicates "
                  "merge-induced decay.", "tab:ablation")
    print("\n" + tab.to_string(index=False))
    return tab


if __name__ == "__main__":
    run()
