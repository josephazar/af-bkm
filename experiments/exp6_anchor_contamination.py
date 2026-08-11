"""Experiment 6 - anchor-contamination stress test.

Threat model: the trusted benign anchor is collected once during commissioning.
An attacker already active in that window can place attack traffic inside the
2,000-sample baseline. Contamination is applied before fitting, so the scaler,
mean, covariance, threshold, and coordinator anchor all inherit it.
"""
import numpy as np
import pandas as pd
from afbkm import datasets, federated as F
import common as C

SEEDS = [0, 1, 2, 3, 4]
LEVELS = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]
SIM = dict(n_workers=3, n_baseline=2000, window=1000, merges=(6, 12, 18),
           alpha=0.5, max_epochs=24, blend=0.5)
CFG = {
    "Original": dict(aggregation="original", threshold_strategy="percentile",
                     threshold_kwargs={"percentile": 90}),
    "AF-BKM": dict(aggregation="robust", consensus_mode="median", trust_k=3.0,
                   threshold_strategy="mad", threshold_kwargs={"k": 3.0}),
}
METRICS = ["precision", "recall", "f1", "fpr"]
COLORS = {"Original": "#d62728", "AF-BKM": "#2ca02c"}


def _final(df):
    values = df.groupby("epoch")[METRICS].mean().tail(3).mean()
    return {metric: float(values[metric]) for metric in METRICS}


def run():
    rows = []
    for dataset in ["nsl-kdd", "unsw-nb15"]:
        X, y, _ = datasets.load(dataset)
        for fraction in LEVELS:
            for method, config in CFG.items():
                collected = {metric: [] for metric in METRICS}
                for seed in SEEDS:
                    frame = F.run_simulation(
                        X, y, seed=seed, contaminate_baseline=fraction,
                        label=f"contam{fraction}", **config, **SIM
                    )
                    final = _final(frame)
                    for metric in METRICS:
                        collected[metric].append(final[metric])
                rows.append(dict(
                    dataset=dataset, contamination=fraction, method=method,
                    **{metric: np.mean(collected[metric]) for metric in METRICS},
                    **{metric + "_std": np.std(collected[metric]) for metric in METRICS},
                ))
            print(f"[exp6] {dataset} contamination={fraction:.2f} done")

    results = pd.DataFrame(rows)
    results.to_csv(f"{C.TAB}/exp6_anchor_contamination.csv", index=False)

    table_rows = []
    for dataset in ["nsl-kdd", "unsw-nb15"]:
        for fraction in LEVELS:
            row = results[
                (results.dataset == dataset)
                & (results.contamination == fraction)
                & (results.method == "AF-BKM")
            ].iloc[0]
            table_rows.append({
                "Dataset": C.DATASET_TITLE[dataset],
                "Contamination": f"{100 * fraction:.0f}%",
                "Precision": f"{row.precision:.3f} {C.PM} {row.precision_std:.3f}",
                "Recall": f"{row.recall:.3f} {C.PM} {row.recall_std:.3f}",
                "F1": f"{row.f1:.3f} {C.PM} {row.f1_std:.3f}",
                "FPR": f"{row.fpr:.3f} {C.PM} {row.fpr_std:.3f}",
            })
    C.write_table(
        pd.DataFrame(table_rows), "exp6_anchor_contamination",
        "AF-BKM under commissioning-anchor contamination: a fraction $p$ of the "
        "nominally benign collection window is replaced with attack traffic "
        "(mean$\\pm$std over 5 seeds).",
        "tab:contamination",
    )

    fig, axes = C.plt.subplots(2, 2, figsize=(12.5, 8))
    for column, dataset in enumerate(["nsl-kdd", "unsw-nb15"]):
        for row_index, metric in enumerate(["f1", "fpr"]):
            ax = axes[row_index, column]
            for method in CFG:
                subset = results[
                    (results.dataset == dataset) & (results.method == method)
                ].sort_values("contamination")
                ax.errorbar(
                    100 * subset.contamination, subset[metric],
                    yerr=subset[metric + "_std"], marker="o", lw=2,
                    capsize=3, color=COLORS[method], label=method,
                )
            ax.set_xlabel("anchor contamination (%)")
            ax.set_ylabel(metric.upper() if metric == "fpr" else "F1-score")
            if row_index == 0:
                ax.set_title(C.DATASET_TITLE[dataset])
            ax.legend()
    fig.suptitle("Sensitivity to contamination of the commissioning anchor", y=1.0)
    C.savefig(fig, "exp6_anchor_contamination")
    print(results.round(3).to_string(index=False))


if __name__ == "__main__":
    run()
