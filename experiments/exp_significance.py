"""Paired significance tests for the central claim (Baseline vs AF-BKM), using the
per-seed results saved by exp1. Reports Wilcoxon signed-rank p, a bootstrap 95% CI
of the mean paired difference, and the effect size (Cohen's d_z).
Run exp1_ablation.py first (it writes exp1_ablation_raw.csv).
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import common as C

BASE = "Baseline (original)"
AFBKM = "AF-BKM (E1+E2)"


def _per_seed(raw, ds, cfg):
    g = raw[(raw.dataset == ds) & (raw.config == cfg)]
    pe = g.groupby(["seed", "epoch"])[["precision", "fpr"]].mean().reset_index()
    out = {"final_precision": {}, "mean_fpr": {}, "delta_precision": {}}
    for seed, s in pe.groupby("seed"):
        s = s.sort_values("epoch")
        out["final_precision"][seed] = s.precision.tail(3).mean()
        out["mean_fpr"][seed] = s.fpr.mean()
        out["delta_precision"][seed] = s.precision.tail(3).mean() - s.precision.head(3).mean()
    return out


def _boot_ci(diff, n=10000):
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def run():
    raw = pd.read_csv(f"{C.TAB}/exp1_ablation_raw.csv")
    rows = []
    # metric, higher-is-better orientation for the difference AF-BKM vs Baseline
    specs = [("final_precision", "Final precision", +1),
             ("mean_fpr", "Mean FPR", -1),                 # improvement = reduction
             ("delta_precision", f"Precision decay {C.DELTA}P", +1)]
    for ds in ["nsl-kdd", "unsw-nb15", "n-baiot"]:
        b = _per_seed(raw, ds, BASE)
        a = _per_seed(raw, ds, AFBKM)
        seeds = sorted(b["final_precision"])
        for key, label, sign in specs:
            bv = np.array([b[key][s] for s in seeds])
            av = np.array([a[key][s] for s in seeds])
            diff = sign * (av - bv)                          # >0 means AF-BKM better
            lo, hi = _boot_ci(diff)
            try:
                p = wilcoxon(av, bv).pvalue
            except ValueError:
                p = float("nan")
            dz = float(diff.mean() / (diff.std(ddof=1) + 1e-12))
            rows.append({
                "Dataset": C.DATASET_TITLE[ds], "Metric": label,
                "Baseline": f"{bv.mean():.3f} {C.PM} {bv.std():.3f}",
                "AF-BKM": f"{av.mean():.3f} {C.PM} {av.std():.3f}",
                "Improvement [95% CI]": f"{diff.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]",
                "Wilcoxon p": f"{p:.4f}", "Cohen dz": f"{dz:+.2f}",
            })
    tab = pd.DataFrame(rows)
    C.write_table(tab, "exp_significance",
                  f"Paired comparison of AF-BKM vs.\\ the original Baseline across {len(seeds)} seeds. "
                  "Improvement is the mean paired difference (FPR oriented as a reduction) with a "
                  "bootstrap 95\\% CI; significance by Wilcoxon signed-rank; effect size as Cohen's $d_z$.",
                  "tab:significance", col_align="l l c c c c c")
    print(tab.to_string(index=False))


if __name__ == "__main__":
    run()
