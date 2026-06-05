# AF-BKM — Adaptive Federated Baseline K-Means for Lightweight IoT Intrusion Detection

A lightweight, **statistics-only**, semi-supervised **federated** intrusion-detection method
for IoT networks. AF-BKM removes the *merge-induced precision / false-positive decay* of the
federated Baseline K-Means detector through two label-free, statistics-only changes:

- **E1 — adaptive auto-thresholding** — a tuning-free decision threshold read from the benign
  Mahalanobis-distance distribution (robust MAD, or a target-FPR calibration). Per-worker
  threshold candidates are used only for outlier-worker filtering.
- **E2 — robust, benign-anchored aggregation** — count/quality-weighted blending of worker
  means with a trusted benign anchor and a trust gate; the global threshold is re-calibrated
  on the trusted benign anchor instead of being tightened toward the nearest anomaly.

"Label-free" means *with respect to attack labels*: like any semi-supervised novelty detector,
a small trusted benign set is assumed for benign-only calibration.

## Results (baseline-only scaling, 10 seeds, non-IID)

| Metric (Baseline → AF-BKM) | NSL-KDD | UNSW-NB15 |
|---|---|---|
| Precision-decay ΔP (first→last epochs) | −0.134 → **−0.002** | −0.121 → **−0.014** |
| Mean FPR | 0.208 → **0.145** | 0.219 → **0.079** |
| Final precision | 0.692 → **0.792** | 0.692 → **0.808** |

All central improvements are significant (paired Wilcoxon *p* = 0.002 across 10 seeds; bootstrap
95% CIs exclude 0; Cohen *d_z* +2.2…+6.0). AF-BKM preserves recall on NSL-KDD and exposes an
explicit precision–recall trade-off on UNSW-NB15 via a benign target-FPR knob; it is competitive
with and cheaper than classical detectors (fit ~4 ms; ~35 ms to score 20k packets) and shares
~90% fewer values per round than covariance-sharing schemes.

## Install & run

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
export PYTHONPATH="$PWD:$PWD/experiments"
./.venv/bin/python -m afbkm.datasets               # build/cache datasets
./.venv/bin/python experiments/exp1_ablation.py    # ablation (the central result)
# ... other experiments under experiments/, or run everything:
bash scripts/reproduce_all.sh
```

Everything runs in `.venv` only and is seeded.

## Layout

```
afbkm/        datasets, thresholds (E1), core (Mahalanobis), baselines, federated (E2 + simulation), metrics
experiments/  exp1_ablation, exp_significance, exp1b_threshold_sensitivity, exp2_sweeps,
              exp3_baselines, exp4_comms_poison, exp5_target_fpr, common.py
scripts/      reproduce_all.sh
```

Running the experiments writes figures and tables to a local `results/` directory
(generated at runtime; not tracked).

## Datasets

- **NSL-KDD** — `defcom17/NSL_KDD` mirror (`KDDTrain+`, `KDDTest+`).
- **UNSW-NB15** — canonical partitioned CSVs (45 columns incl. `attack_cat`, `label`).

Both are loaded and cached by `afbkm/datasets.py`; large raw files are git-ignored and fetched
locally. Scaling is fit on the benign baseline only (no preprocessing leakage).

## License

MIT — see [LICENSE](LICENSE).
