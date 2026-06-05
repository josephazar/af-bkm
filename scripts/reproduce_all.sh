#!/usr/bin/env bash
# Reproduce all AF-BKM experiments (figures and tables are written to results/).
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"

# Environment (venv only)
[ -d .venv ] || python3 -m venv .venv
./.venv/bin/pip install -q -r requirements.txt
export PYTHONPATH="$ROOT:$ROOT/experiments"

# Datasets (no-leakage scaling is applied inside the experiments)
./.venv/bin/python -m afbkm.datasets

# Experiments (exp1 must precede exp_significance, which reads its raw CSV)
for e in exp1_ablation exp_significance exp1b_threshold_sensitivity \
         exp2_sweeps exp3_baselines exp4_comms_poison exp5_target_fpr; do
  echo ">>> running $e"
  ./.venv/bin/python "experiments/$e.py"
done

echo "Done -> results/ (figures, tables)"
