# AF-BKM: Adaptive Federated Baseline K-Means for Lightweight IoT Intrusion Detection

AF-BKM is a statistics-only, semi-supervised federated intrusion detector. It
extends the Baseline K-Means Mahalanobis detector with:

- **E1, adaptive thresholding:** a threshold estimated from benign distances
  with MAD or explicit target-FPR calibration. Worker candidates are used only
  by the update trust gate.
- **E2, benign-anchored aggregation:** count/quality-weighted worker means are
  blended with a commissioning-time benign anchor, and the global threshold is
  recalibrated on that anchor after each merge.

"Label-free" refers to attack labels during operation. As in other one-class
detectors, AF-BKM assumes a small nominally benign commissioning set.

The method is described in [*IoT* **2026**, 7(3), 67](https://doi.org/10.3390/iot7030067)
(open access, CC BY 4.0). See [Citation](#citation).

## Main result

The central non-IID ablation uses 10 seeds, three workers, and three merge rounds.

| Metric (original to AF-BKM) | NSL-KDD | UNSW-NB15 | N-BaIoT |
|---|---:|---:|---:|
| Precision change, early to late | -0.134 to **-0.002** | -0.121 to **-0.014** | -0.170 to **-0.009** |
| Mean FPR | 0.203 to **0.144** | 0.215 to **0.078** | 0.199 to **0.105** |
| Final precision | 0.692 to **0.792** | 0.692 to **0.808** | 0.681 to **0.838** |

These improvements are paired by seed. The repository also contains threshold
sensitivity, non-IID/worker-count sweeps, detector comparisons, selected
faulty-worker stress tests, commissioning-anchor contamination, desktop resource
profiling, and white-box feature-space evasion experiments.

## Environment

The locked environment was generated with Python 3.12. Create it with:

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.lock.txt
export PYTHONPATH="$PWD:$PWD/experiments"
```

Use `requirements.txt` instead of `requirements.lock.txt` when testing compatible
newer dependency versions.

## Dataset preparation

Raw data and generated caches are intentionally excluded from Git. By default,
the loader uses `data/`; set `AFBKM_DATA_DIR` to use another location:

```bash
export AFBKM_DATA_DIR="$PWD/data"
mkdir -p "$AFBKM_DATA_DIR/raw"
```

Stage these exact inputs before running the experiments.

### NSL-KDD

Place `KDDTrain+.txt` and `KDDTest+.txt` under
`$AFBKM_DATA_DIR/raw/nsl-kdd/`. They are available from the
[NSL-KDD repository](https://github.com/defcom17/NSL_KDD). The pooled corpus
must produce 148,517 rows (77,054 benign and 71,463 attack) and 37 retained
numeric features.

### UNSW-NB15

Download the official training and testing CSVs from the
[UNSW-NB15 dataset page](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
and place them under `$AFBKM_DATA_DIR/raw/unsw-nb15/` as
`UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv`. The legacy names
`UNSW_NB15_partA.csv` and `UNSW_NB15_partB.csv` are also accepted. The pooled
corpus must produce 257,673 rows (93,000 benign and 164,673 attack) and 39
retained numeric features.

### N-BaIoT

The experiments use the prepared 50-client non-IID partition distributed by
[FedMSE at commit d126885](https://github.com/dino-chiio/fedmse/tree/d1268850696ffa7e9ca5469c0fc9276e966eafd7),
not a fresh resampling of the UCI files. Reconstruct its split archive and stage
only that scenario:

```bash
git clone https://github.com/dino-chiio/fedmse.git /tmp/fedmse
git -C /tmp/fedmse checkout d1268850696ffa7e9ca5469c0fc9276e966eafd7
cd /tmp/fedmse/Data
zip -s 0 Prepared_dataset.zip --out Prepared_dataset-full.zip
unzip Prepared_dataset-full.zip

cd /path/to/af_bkm_paper
export AFBKM_DATA_DIR="$PWD/data"
mkdir -p "$AFBKM_DATA_DIR/raw/n-baiot/nonIID-50"
cp -R /tmp/fedmse/Data/Scenarios_dataset/Scenarios_dataset/nonIID-50-Client_Data/Client-* \
  "$AFBKM_DATA_DIR/raw/n-baiot/nonIID-50/"
```

The loader requires all 50 client directories and validates the published corpus:
136,565 rows, 115 features, 71,716 benign samples, and 64,849 attack samples.
It pools the prepared clients; each experiment then applies its documented,
seeded federated partitioning.

Build the local caches and verify all three summaries:

```bash
cd /path/to/af_bkm_paper
export PYTHONPATH="$PWD:$PWD/experiments"
./.venv/bin/python -m afbkm.datasets
```

Published experiments cache unscaled arrays and fit MinMax scaling at runtime,
on the commissioning baseline, applying it to future stream samples. The
near-constant feature filter is the disclosed exception: it is label-blind
full-corpus preprocessing performed before the seeded experiment splits.
The optional `scale="global"` appendix path uses a separately named,
digest-validated scaled cache.

Every cache is content-verified on load. Unscaled caches are hashed against a
pinned sha256 of the published corpus, which fixes row order as well as values;
scaled caches additionally carry the digest of the raw corpus they derive from
and of their own contents. A mismatch raises instead of silently returning
different data. `python -m afbkm.datasets` prints the digests.

## Reproduce

Run the fast integration check:

```bash
PYTHONPATH="$PWD:$PWD/experiments" ./.venv/bin/python experiments/smoke_test.py
```

Run the complete experiment suite:

```bash
bash scripts/reproduce_all.sh
```

`exp1_ablation` must precede `exp_significance`; the run script enforces this.
Figures, CSV files, and generated LaTeX tables are written to the ignored
`results/` directory. Resource timings identify the benchmark CPU and report
Python-traced peak allocations. They are not hardware deployment or energy
measurements.

## Layout

```text
afbkm/        detector, thresholds, aggregation, datasets, metrics, comparators
experiments/  exp1 ablation and significance; exp1b threshold sensitivity;
              exp2 sweeps; exp3 comparators; exp4 communication/faulty updates;
              exp5 target FPR; exp6 anchor contamination; exp7 resources;
              exp8 white-box evasion; smoke test
scripts/      complete reproduction entry point
```

## Scope

The faulty-worker experiments evaluate the explicit update manipulations in
`exp4_comms_poison.py`; they do not establish worst-case Byzantine robustness.
The resource experiment is a desktop benchmark. Physical edge-device latency,
energy, live-network behavior, privacy leakage, and broader adaptive attacks are
outside this code release's empirical claims.

## Citation

If this code is useful in your work, please cite the paper:

> Al Saleh, M.; Azar, J. Adaptive Federated Baseline K-Means for Lightweight IoT
> Intrusion Detection: Auto-Thresholding and Robust Statistics Aggregation.
> *IoT* **2026**, *7*(3), 67. https://doi.org/10.3390/iot7030067

```bibtex
@Article{iot7030067,
AUTHOR = {Al Saleh, Mohammed and Azar, Joseph},
TITLE = {Adaptive Federated Baseline K-Means for Lightweight IoT Intrusion Detection: Auto-Thresholding and Robust Statistics Aggregation},
JOURNAL = {IoT},
VOLUME = {7},
YEAR = {2026},
NUMBER = {3},
ARTICLE-NUMBER = {67},
URL = {https://www.mdpi.com/2624-831X/7/3/67},
ISSN = {2624-831X},
DOI = {10.3390/iot7030067}
}
```

The article is open access under a Creative Commons Attribution (CC BY 4.0)
licence. The code in this repository is released separately under the MIT
licence; see below.

## License

MIT; see [LICENSE](LICENSE).
