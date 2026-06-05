"""Experiment 4 - (a) communication cost (statistics-only vs sharing covariance)
and (b) sensitivity to several poisoning attacks, comparing the original merge with
AF-BKM. Honest-worker precision AND recall are reported.
"""
import numpy as np
import pandas as pd
from afbkm import datasets, federated as F, metrics as M
import common as C

SEEDS = [0, 1, 2, 3, 4]
SIM = dict(n_baseline=2000, window=1000, merges=(6, 12, 18), alpha=0.5,
           max_epochs=24, blend=0.5)
ORIG = dict(aggregation="original", threshold_strategy="percentile",
            threshold_kwargs={"percentile": 90})
AF = dict(aggregation="robust", consensus_mode="median", trust_k=3.0,
          threshold_strategy="mad", threshold_kwargs={"k": 3.0})
# (label, poison_kind, poisoned worker ids, n_workers)
SCENARIOS = [
    ("Clean", "evasion", (), 3),
    ("Evasion 1/3", "evasion", (0,), 3),
    ("Mean-shift 1/3", "mean_shift", (0,), 3),
    ("Thr-inflate 1/3", "thr_high", (0,), 3),
    ("Evasion 2/5", "evasion", (0, 1), 5),
]


def _final(df, col):
    return df.groupby("epoch")[col].mean().tail(3).mean()


def run():
    # ---------- (a) communication cost ----------
    rows = []
    for ds, d in [("nsl-kdd", 37), ("unsw-nb15", 39)]:
        so = M.comms_floats_per_round(d, share_cov=False)
        cv = M.comms_floats_per_round(d, share_cov=True)
        rows.append({"Dataset": C.DATASET_TITLE[ds], "d": d,
                     "Statistics-only (floats)": so, "Share covariance (floats)": cv,
                     "Uplink reduction": f"{100 * (1 - so / cv):.1f}%",
                     "Stats (B/round)": so * 4, "Cov (B/round)": cv * 4})
    comms = pd.DataFrame(rows)
    C.write_table(comms, "exp4_comms",
                  "Per-worker uplink per merge round (fp32). AF-BKM transmits only means and a "
                  "few scalars; sharing the covariance grows quadratically in $d$.", "tab:comms")
    print(comms.to_string(index=False))

    # ---------- (b) poisoning sensitivity (NSL-KDD) ----------
    X, y, _ = datasets.load("nsl-kdd")
    prows = []
    for name, kind, poison, nw in SCENARIOS:
        for dname, cfg in [("Original", ORIG), ("AF-BKM", AF)]:
            P, R = [], []
            for s in SEEDS:
                df = F.run_simulation(X, y, n_workers=nw, seed=s, poison_workers=poison,
                                      poison_kind=kind, label=name, **cfg, **SIM)
                honest = df[~df.worker.isin({f"W{i}" for i in poison})] if poison else df
                P.append(_final(honest, "precision")); R.append(_final(honest, "recall"))
            prows.append(dict(scenario=name, defense=dname,
                              precision=np.mean(P), precision_std=np.std(P),
                              recall=np.mean(R), recall_std=np.std(R)))
        print(f"[exp4] scenario '{name}' done")
    P = pd.DataFrame(prows)
    P.to_csv(f"{C.TAB}/exp4_poison.csv", index=False)

    # table
    piv = P.pivot(index="scenario", columns="defense")
    order = [s[0] for s in SCENARIOS]
    tab = pd.DataFrame({"Scenario": order})
    tab["Orig. P"] = [f"{piv['precision']['Original'][s]:.3f}" for s in order]
    tab["Orig. R"] = [f"{piv['recall']['Original'][s]:.3f}" for s in order]
    tab["AF-BKM P"] = [f"{piv['precision']['AF-BKM'][s]:.3f}" for s in order]
    tab["AF-BKM R"] = [f"{piv['recall']['AF-BKM'][s]:.3f}" for s in order]
    C.write_table(tab, "exp4_poison",
                  "Honest-worker precision (P) and recall (R) under poisoning (mean over 5 seeds). "
                  "AF-BKM is markedly less sensitive than the original merge to threshold-inflation "
                  "attacks (evasion, thr-inflate); mean-shift degrades it gracefully but not fully.",
                  "tab:poison")

    # figure: recall by scenario, two defenses
    fig, axes = C.plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, lab in zip(axes, ["precision", "recall"], ["Precision", "Recall"]):
        xpos = np.arange(len(order)); w = 0.38
        for k, dname in enumerate(["Original", "AF-BKM"]):
            vals = [piv[metric][dname][s] for s in order]
            errs = [P[(P.scenario == s) & (P.defense == dname)][f"{metric}_std"].values[0] for s in order]
            ax.bar(xpos + (k - 0.5) * w, vals, width=w, yerr=errs, capsize=3,
                   label=dname, color=("#d62728" if k == 0 else "#2ca02c"))
        ax.set_xticks(xpos); ax.set_xticklabels(order, rotation=15, fontsize=11)
        ax.set_ylabel(f"honest-worker {lab.lower()}"); ax.set_ylim(0, 1)
        ax.set_title(lab); ax.legend()
    fig.suptitle("Sensitivity to poisoning attacks (NSL-KDD): AF-BKM vs. original merge", y=1.02)
    C.savefig(fig, "exp4_poison")
    print(tab.to_string(index=False))


if __name__ == "__main__":
    run()
