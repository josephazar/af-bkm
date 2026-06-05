"""AF-BKM: Adaptive Federated Baseline K-Means for lightweight IoT intrusion detection.

Modules:
  datasets   - NSL-KDD / UNSW-NB15 loading + common preprocessing
  thresholds - E1: adaptive threshold strategies + federated threshold consensus
  core       - Mahalanobis novelty model (vectorised) + baseline fitting
  baselines  - sklearn novelty-detector baselines (OCSVM/IF/LOF/KDE/KNN)
  federated  - workers, coordinator, aggregation modes (E2), simulation harness
  metrics    - precision/recall/F1/F2/FPR/AUC + communication cost
"""
__version__ = "0.1.0"
