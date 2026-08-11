"""Experiment 8 - white-box feature-space evasion.

The attacker knows the benign centroid and moves each attack sample toward it:
``x' = x + t * (mu - x)`` for ``t`` in [0, 1). Mahalanobis distance is
homogeneous around the mean, so ``D(x') = (1-t) * D(x)``. The reported scalar is
the budget required to halve each seed's unperturbed recall, not the budget for
an absolute recall of 0.5.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from afbkm import datasets, core
import common as C

SEEDS = [0, 1, 2, 3, 4]
N_BASE = 2000
N_EVAL = 20000
BUDGETS = np.linspace(0.0, 0.995, 200)
STRATEGIES = [
    ("percentile p=90", "percentile", {"percentile": 90}),
    ("MAD k=3", "mad", {"k": 3.0}),
    ("calibrated FPR 5%", "calibrated_quantile", {"target_fpr": 0.05}),
]
COLORS = {0: "#444444", 1: "#2ca02c", 2: "#1f77b4"}


def _halving_budget(attack_distances, threshold, initial_recall):
    if initial_recall <= 0.0:
        return np.nan
    target_recall = 0.5 * initial_recall
    target_distance = np.quantile(attack_distances, 1.0 - target_recall)
    return float(np.clip(1.0 - threshold / target_distance, 0.0, 1.0))


def run():
    rows, curves = [], {}
    for dataset in ["nsl-kdd", "unsw-nb15", "n-baiot"]:
        X, y, _ = datasets.load(dataset)
        curves[dataset] = {}
        for strategy_index, (label, strategy, kwargs) in enumerate(STRATEGIES):
            recall = np.zeros((len(SEEDS), len(BUDGETS)))
            half_budgets = []
            for seed_index, seed in enumerate(SEEDS):
                rng = np.random.default_rng(seed)
                permutation = rng.permutation(len(X))
                shuffled_X, shuffled_y = X[permutation], y[permutation]
                baseline_idx = np.where(shuffled_y == 0)[0][:N_BASE]
                stream_mask = np.ones(len(shuffled_X), dtype=bool)
                stream_mask[baseline_idx] = False
                scaler = MinMaxScaler().fit(shuffled_X[baseline_idx])
                model, _ = core.fit_baseline(
                    scaler.transform(shuffled_X[baseline_idx]),
                    threshold_strategy=strategy,
                    threshold_kwargs=kwargs,
                )
                eval_X = scaler.transform(shuffled_X[stream_mask][:N_EVAL])
                eval_y = shuffled_y[stream_mask][:N_EVAL]
                attack_distances = model.distance(eval_X[eval_y == 1])
                recall[seed_index] = [
                    (attack_distances > model.threshold / (1.0 - budget)).mean()
                    for budget in BUDGETS
                ]
                half_budgets.append(_halving_budget(
                    attack_distances, model.threshold, recall[seed_index, 0]
                ))
            curves[dataset][label] = (recall.mean(axis=0), recall.std(axis=0))
            rows.append({
                "Dataset": C.DATASET_TITLE[dataset],
                "Threshold rule": label,
                "Initial recall": f"{recall[:, 0].mean():.3f}",
                "Budget to halve recall": (
                    f"{np.nanmean(half_budgets):.2f} {C.PM} {np.nanstd(half_budgets):.2f}"
                ),
            })
        print(f"[exp8] {dataset} done")

    table = pd.DataFrame(rows)
    C.write_table(
        table, "exp8_evasion",
        "White-box feature-space evasion toward the benign centroid. Budget $t$ "
        "is the fraction of the path from an attack sample to $\\mu$; the table "
        "reports the budget required to halve unperturbed recall "
        "(mean$\\pm$std over 5 seeds). Larger is better.",
        "tab:evasion",
    )

    fig, axes = C.plt.subplots(1, 3, figsize=(15.5, 4.6))
    for ax, dataset in zip(axes, ["nsl-kdd", "unsw-nb15", "n-baiot"]):
        for strategy_index, (label, _, _) in enumerate(STRATEGIES):
            mean, std = curves[dataset][label]
            ax.plot(BUDGETS, mean, lw=2, color=COLORS[strategy_index], label=label)
            ax.fill_between(BUDGETS, mean - std, mean + std,
                            color=COLORS[strategy_index], alpha=0.15)
        ax.set_xlabel("evasion budget $t$ (fraction of path to $\\mu$)")
        ax.set_ylabel("recall on perturbed attacks")
        ax.set_title(C.DATASET_TITLE[dataset])
        ax.set_ylim(0, 1)
        ax.legend(fontsize=10)
    fig.suptitle("Recall under white-box perturbation toward the benign centroid", y=1.03)
    C.savefig(fig, "exp8_evasion")
    print(table.to_string(index=False))


if __name__ == "__main__":
    run()
