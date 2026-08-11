"""Experiment 7 - computational resource profile.

Timing and Python-traced allocation are measured in separate passes because
``tracemalloc`` changes execution time. Model and communication footprints are
closed-form fp32 counts. This script intentionally does not infer target-device
energy from workstation timing; energy requires measurement on the target.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import pickle
import platform
import subprocess
import time
import tracemalloc

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from afbkm import baselines, core, datasets, metrics as M
import common as C

SEEDS = [0, 1, 2]
N_BASE = 2000
N_EVAL = 20000
N_SINGLE = 2000


def cpu_name():
    try:
        name = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if name:
            return name
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or platform.machine() or "unknown CPU"


def bkm_model_bytes(d):
    """Deployment serialization: mean, inverse covariance, and two fp32 scalars."""
    return (d + d * d + 2) * 4


def _mean_std(values, digits=1):
    values = np.asarray(values, dtype=float)
    return f"{values.mean():.{digits}f} {C.PM} {values.std():.{digits}f}"


def profile_bkm(baseline_X, eval_X):
    timing = {}
    start = time.perf_counter()
    model, _ = core.fit_baseline(
        baseline_X, threshold_strategy="mad", threshold_kwargs={"k": 3.0}
    )
    timing["fit_ms"] = 1e3 * (time.perf_counter() - start)

    start = time.perf_counter()
    model.predict(eval_X)
    timing["batch_ms"] = 1e3 * (time.perf_counter() - start)

    start = time.perf_counter()
    for index in range(N_SINGLE):
        model.predict(eval_X[index:index + 1])
    timing["per_packet_us"] = 1e6 * (time.perf_counter() - start) / N_SINGLE

    tracemalloc.start()
    core.fit_baseline(
        baseline_X, threshold_strategy="mad", threshold_kwargs={"k": 3.0}
    )
    timing["fit_peak_kb"] = tracemalloc.get_traced_memory()[1] / 1024
    tracemalloc.stop()

    tracemalloc.start()
    model.predict(eval_X)
    timing["score_peak_kb"] = tracemalloc.get_traced_memory()[1] / 1024
    tracemalloc.stop()
    return timing


def _split_and_scale(X, y, seed):
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(X))
    shuffled_X, shuffled_y = X[permutation], y[permutation]
    baseline_idx = np.where(shuffled_y == 0)[0][:N_BASE]
    stream_mask = np.ones(len(shuffled_X), dtype=bool)
    stream_mask[baseline_idx] = False
    scaler = MinMaxScaler().fit(shuffled_X[baseline_idx])
    return (
        scaler.transform(shuffled_X[baseline_idx]),
        scaler.transform(shuffled_X[stream_mask][:N_EVAL]),
    )


def run():
    cpu = cpu_name()
    print(f"[exp7] benchmark CPU: {cpu}")

    worker_rows = []
    for dataset in ["nsl-kdd", "unsw-nb15", "n-baiot"]:
        X, y, _ = datasets.load(dataset)
        profiles = []
        for seed in SEEDS:
            baseline_X, eval_X = _split_and_scale(X, y, seed)
            profiles.append(profile_bkm(baseline_X, eval_X))
        values = {
            key: [profile[key] for profile in profiles]
            for key in profiles[0]
        }
        d = X.shape[1]
        worker_rows.append({
            "Dataset": C.DATASET_TITLE[dataset],
            "d": d,
            "Model (KB)": f"{bkm_model_bytes(d) / 1024:.1f}",
            "Uplink (B/round)": M.comms_floats_per_round(d) * 4,
            "Fit (ms)": _mean_std(values["fit_ms"]),
            "Score 20k (ms)": _mean_std(values["batch_ms"]),
            "Per-packet (us)": _mean_std(values["per_packet_us"]),
            "Peak traced fit (KB)": _mean_std(values["fit_peak_kb"], digits=0),
        })
        print(f"[exp7] AF-BKM on {dataset} done")

    worker_table = pd.DataFrame(worker_rows)
    C.write_table(
        worker_table, "exp7_resources_bkm",
        "AF-BKM worker-side resource profile on the three datasets "
        f"(benchmark CPU: {cpu}; mean$\\pm$std over 3 seeds). Timings are desktop "
        "benchmarks, model and uplink sizes are closed-form fp32 values, and peak "
        "allocation is Python-traced rather than total process memory.",
        "tab:resources_bkm",
    )

    X, y, _ = datasets.load("nsl-kdd")
    baseline_X, eval_X = _split_and_scale(X, y, seed=0)
    detector_rows = []
    bkm_profile = profile_bkm(baseline_X, eval_X)
    detector_rows.append({
        "Detector": "Bkmeans-Mahalanobis",
        "Model (KB)": f"{bkm_model_bytes(X.shape[1]) / 1024:.1f}",
        "Fit (ms)": f"{bkm_profile['fit_ms']:.1f}",
        "Score 20k (ms)": f"{bkm_profile['batch_ms']:.1f}",
        "Per-packet (us)": f"{bkm_profile['per_packet_us']:.1f}",
        "Peak traced score (KB)": f"{bkm_profile['score_peak_kb']:.0f}",
    })
    for name, detector in baselines.make_detectors(seed=0).items():
        start = time.perf_counter()
        detector.fit(baseline_X)
        fit_ms = 1e3 * (time.perf_counter() - start)

        start = time.perf_counter()
        detector.predict(eval_X)
        batch_ms = 1e3 * (time.perf_counter() - start)

        start = time.perf_counter()
        for index in range(200):
            detector.predict(eval_X[index:index + 1])
        per_packet_us = 1e6 * (time.perf_counter() - start) / 200

        tracemalloc.start()
        detector.predict(eval_X)
        score_peak_kb = tracemalloc.get_traced_memory()[1] / 1024
        tracemalloc.stop()
        detector_rows.append({
            "Detector": name,
            "Model (KB)": f"{len(pickle.dumps(detector)) / 1024:.0f}",
            "Fit (ms)": f"{fit_ms:.1f}",
            "Score 20k (ms)": f"{batch_ms:.1f}",
            "Per-packet (us)": f"{per_packet_us:.1f}",
            "Peak traced score (KB)": f"{score_peak_kb:.0f}",
        })
        print(f"[exp7] {name} done")

    detector_table = pd.DataFrame(detector_rows)
    C.write_table(
        detector_table, "exp7_resources_detectors",
        "Exploratory desktop resource comparison on NSL-KDD "
        f"(benchmark CPU: {cpu}; one seeded split). AF-BKM model size is a "
        "closed-form fp32 serialization; comparator sizes are serialized Python objects. "
        "Peak allocation is Python-traced rather than total process memory.",
        "tab:resources_detectors",
    )
    print(worker_table.to_string(index=False))
    print(detector_table.to_string(index=False))


if __name__ == "__main__":
    run()
