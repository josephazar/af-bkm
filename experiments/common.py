"""Shared paths, plotting style, and helpers for AF-BKM experiments."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "results", "figures")
TAB = os.path.join(ROOT, "results", "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

METRICS = ["precision", "recall", "f1", "f2", "fpr", "accuracy"]

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "font.size": 14,
    "axes.titlesize": 16, "axes.labelsize": 14, "legend.fontsize": 12,
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
    "axes.spines.right": False, "figure.autolayout": True,
})

# colour-blind-friendly palette
PALETTE = {
    "Baseline (original)": "#d62728",
    "E1 adaptive-thr": "#ff7f0e",
    "E2 robust-agg": "#1f77b4",
    "AF-BKM (E1+E2)": "#2ca02c",
}
DATASET_TITLE = {"nsl-kdd": "NSL-KDD", "unsw-nb15": "UNSW-NB15"}


def savefig(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figure: {name}.pdf/.png")


def per_epoch(df):
    """Average raw per-worker rows to one row per (dataset, config, seed, epoch)."""
    keys = ["dataset", "config", "seed", "epoch"]
    agg = {m: "mean" for m in METRICS if m in df.columns}
    agg["threshold"] = "mean"
    agg["is_merge_epoch"] = "max"
    return df.groupby(keys, as_index=False).agg(agg)


def mean_std_over_seeds(pe):
    """Mean and std across seeds -> one row per (dataset, config, epoch)."""
    keys = ["dataset", "config", "epoch"]
    cols = [m for m in METRICS if m in pe.columns] + ["threshold"]
    g = pe.groupby(keys)[cols].agg(["mean", "std"]).reset_index()
    g.columns = ["_".join(c).rstrip("_") for c in g.columns]
    merge_ep = pe.groupby(keys)["is_merge_epoch"].max().reset_index()
    return g.merge(merge_ep, on=keys)


_TEX_REPL = {"\\": r"\textbackslash{}", "_": r"\_", "%": r"\%", "&": r"\&",
             "#": r"\#", "±": r"$\pm$", "Δ": r"$\Delta$", "→": r"$\rightarrow$",
             "≥": r"$\geq$", "≤": r"$\leq$", "α": r"$\alpha$"}


def _tex(s):
    s = str(s)
    for k, v in _TEX_REPL.items():
        s = s.replace(k, v)
    return s


def write_table(df, name, caption, label, col_align=None):
    """Dependency-free booktabs LaTeX table + CSV (avoids pandas/jinja2 Styler)."""
    df.to_csv(os.path.join(TAB, f"{name}.csv"), index=False)
    cols = list(df.columns)
    align = col_align or ("l" + "c" * (len(cols) - 1))
    lines = ["\\begin{table}[H]", "\\centering", "\\afbkmTableSetup",
             f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\begin{adjustbox}{max width=\\linewidth,center}",
             f"\\begin{{tabular}}{{{align}}}", "\\toprule",
             " & ".join(_tex(c) for c in cols) + " \\\\", "\\midrule"]
    for _, r in df.iterrows():
        lines.append(" & ".join(_tex(v) for v in r.tolist()) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}", "\\end{table}", ""]
    with open(os.path.join(TAB, f"{name}.tex"), "w") as fh:
        fh.write("\n".join(lines))
    print(f"  saved table: {name}.csv/.tex")
