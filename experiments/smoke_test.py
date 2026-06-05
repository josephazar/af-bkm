"""Smoke test: does the original merge decay precision, and does E1+E2 fix it?"""
import numpy as np
from afbkm import datasets, federated as F

X, y, names = datasets.load("nsl-kdd")
print(f"NSL-KDD: X={X.shape} attack_ratio={(y==1).mean():.3f}")

common = dict(n_workers=3, n_baseline=2000, window=1000,
              merges=(5, 10, 15), alpha=0.5, seed=0, max_epochs=20)


def summarize(df, name):
    g = df.groupby("epoch").agg(P=("precision", "mean"), R=("recall", "mean"),
                                F1=("f1", "mean"), F2=("f2", "mean"),
                                FPR=("fpr", "mean"), thr=("threshold", "mean"))
    print(f"\n=== {name} ===")
    print(g.round(3).to_string())
    return g


df_base = F.run_simulation(X, y, aggregation="original", threshold_strategy="percentile",
                           threshold_kwargs={"percentile": 90}, label="base", **common)
df_e1e2 = F.run_simulation(X, y, aggregation="robust", consensus_mode="median",
                           threshold_strategy="calibrated_quantile",
                           threshold_kwargs={"target_fpr": 0.10}, quality_weight=True,
                           trust_k=3.0, label="E1E2", **common)

gb = summarize(df_base, "ORIGINAL merge (baseline)")
gr = summarize(df_e1e2, "ROBUST E1+E2")
print(f"\nfinal precision : base={gb.P.iloc[-1]:.3f}  E1E2={gr.P.iloc[-1]:.3f}")
print(f"mean  precision : base={gb.P.mean():.3f}  E1E2={gr.P.mean():.3f}")
print(f"mean  recall    : base={gb.R.mean():.3f}  E1E2={gr.R.mean():.3f}")
print(f"mean  F2        : base={gb.F2.mean():.3f}  E1E2={gr.F2.mean():.3f}")
