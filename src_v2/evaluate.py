"""
Evaluation and reproducible inference benchmarking utilities (Version 2).
Strictly distinguishes model inference execution from network packet processing.
Provides macro one-vs-rest multiclass ROC-AUC, per-class report extraction,
and isolated single-flow vs batch latency profiling.
"""

import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_predictions_v2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Computes standard classification metrics with macro OVR ROC-AUC and per-class reports.
    """
    num_classes = len(class_names) if class_names else len(np.unique(y_true))

    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }

    # Macro One-vs-Rest (OVR) ROC-AUC calculation
    metrics["roc_auc_ovr"] = None
    metrics["roc_auc_error"] = None

    if y_prob is not None:
        try:
            if num_classes == 2:
                prob_pos = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
                metrics["roc_auc_ovr"] = float(roc_auc_score(y_true, prob_pos))
            else:
                # Multiclass Macro OVR
                # Note: labels parameter ensures all classes are accounted for
                labels = list(range(num_classes))
                metrics["roc_auc_ovr"] = float(
                    roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro", labels=labels)
                )
        except Exception as e:
            metrics["roc_auc_ovr"] = None
            metrics["roc_auc_error"] = str(e)
    else:
        metrics["roc_auc_error"] = "Model does not produce probability distributions (predict_proba unavailable)."

    # Full Per-Class Report
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    metrics["per_class_report"] = report

    return metrics


def benchmark_single_flow_latency(
    model: Any,
    X_sample: np.ndarray,
    n_samples: int = 2000,
    n_warmup: int = 500,
    n_runs: int = 5,
    scaler: Optional[Any] = None,
) -> Dict[str, float]:
    """
    Measures sequential single-flow inference latency (one sample at a time).
    Simulates real-time streaming arrival: model.predict(x[i:i+1]).
    
    Reports:
      - median_latency_us (P50)
      - p95_latency_us
      - p99_latency_us
      - min_latency_us
      - max_latency_us
    """
    test_subset = X_sample[:n_samples] if len(X_sample) >= n_samples else X_sample
    actual_count = len(test_subset)

    # 1. Warm-up runs (instruction cache priming)
    warmup_n = min(n_warmup, actual_count)
    for i in range(warmup_n):
        x_single = test_subset[i:i+1]
        if scaler is not None:
            x_single = scaler.transform(x_single)
        _ = model.predict(x_single)

    # 2. Sequential measurements across repetitions
    run_latencies_us = []

    for _ in range(n_runs):
        sample_latencies = []
        for i in range(actual_count):
            x_single = test_subset[i:i+1]
            t0 = time.perf_counter()
            if scaler is not None:
                x_single = scaler.transform(x_single)
            _ = model.predict(x_single)
            t1 = time.perf_counter()
            sample_latencies.append((t1 - t0) * 1e6)

        run_latencies_us.extend(sample_latencies)

    arr = np.array(run_latencies_us)

    return {
        "benchmark_type": "single_flow_sequential",
        "preprocessing_included": scaler is not None,
        "n_samples_tested": actual_count,
        "n_warmup_iterations": warmup_n,
        "n_runs": n_runs,
        "median_latency_us": round(float(np.median(arr)), 3),
        "p95_latency_us": round(float(np.percentile(arr, 95)), 3),
        "p99_latency_us": round(float(np.percentile(arr, 99)), 3),
        "min_latency_us": round(float(np.min(arr)), 3),
        "max_latency_us": round(float(np.max(arr)), 3),
    }


def benchmark_batch_throughput(
    model: Any,
    X_sample: np.ndarray,
    batch_size: int = 10000,
    n_warmup: int = 3,
    n_runs: int = 5,
    scaler: Optional[Any] = None,
) -> Dict[str, float]:
    """
    Measures batched inference throughput on in-memory array.
    Reports predictions per second and amortized microseconds per sample.
    """
    test_batch = X_sample[:batch_size] if len(X_sample) >= batch_size else X_sample
    n = len(test_batch)

    # Warm-up
    for _ in range(n_warmup):
        xb = test_batch
        if scaler is not None:
            xb = scaler.transform(xb)
        _ = model.predict(xb)

    batch_durations = []
    for _ in range(n_runs):
        xb = test_batch
        t0 = time.perf_counter()
        if scaler is not None:
            xb = scaler.transform(xb)
        _ = model.predict(xb)
        t1 = time.perf_counter()
        batch_durations.append(t1 - t0)

    median_duration = float(np.median(batch_durations))
    throughput_pps = n / median_duration if median_duration > 0 else 0.0
    amortized_latency_us = (median_duration / n) * 1e6 if n > 0 else 0.0

    return {
        "benchmark_type": "batch_in_memory",
        "batch_size": n,
        "preprocessing_included": scaler is not None,
        "n_runs": n_runs,
        "throughput_predictions_per_sec": round(throughput_pps, 1),
        "amortized_latency_us": round(amortized_latency_us, 3),
    }

