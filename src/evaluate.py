"""
Evaluation utilities: comprehensive metrics, latency profiling, and publication-ready plots.
"""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import FIGURES_DIR, TABLES_DIR


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Computes standard and fine-grained classification metrics.
    """
    num_classes = len(class_names) if class_names else len(np.unique(y_true))

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }

    # AUC calculation if probabilities are provided
    if y_prob is not None:
        try:
            if num_classes == 2:
                # Binary ROC-AUC
                prob_pos = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
                metrics["roc_auc"] = float(roc_auc_score(y_true, prob_pos))
                precision_pts, recall_pts, _ = precision_recall_curve(
                    y_true, prob_pos
                )
                metrics["pr_auc"] = float(auc(recall_pts, precision_pts))
            else:
                # Multi-class One-vs-Rest ROC-AUC
                metrics["roc_auc"] = float(
                    roc_auc_score(
                        y_true, y_prob, multi_class="ovr", average="macro"
                    )
                )
        except Exception as e:
            metrics["roc_auc"] = None
            metrics["pr_auc"] = None

    # Per-class metrics
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    metrics["per_class_report"] = report

    return metrics


def measure_inference_latency(
    model: Any,
    X_sample: np.ndarray,
    n_runs: int = 5,
    n_instances: int = 10_000,
) -> Tuple[float, float]:
    """
    Benchmarks per-sample inference latency (in microseconds) and throughput (flows/sec).
    """
    test_subset = (
        X_sample[:n_instances] if len(X_sample) >= n_instances else X_sample
    )
    actual_count = len(test_subset)

    # Warm-up run
    _ = model.predict(test_subset[:100])

    durations = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = model.predict(test_subset)
        t1 = time.perf_counter()
        durations.append(t1 - t0)

    avg_duration = float(np.median(durations))
    latency_microseconds = (avg_duration / actual_count) * 1_000_000
    throughput_fps = (
        actual_count / avg_duration if avg_duration > 0 else float("inf")
    )

    return latency_microseconds, throughput_fps


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    save_path: Optional[Path] = None,
    normalize: bool = True,
    cmap: str = "Blues",
) -> None:
    """
    Renders and saves a normalized confusion matrix heatmap.
    """
    if normalize:
        cm_norm = (
            cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        )  # Normalized by true class
        cm_display = np.nan_to_num(cm_norm)
        fmt = ".2%"
    else:
        cm_display = cm
        fmt = "d"

    plt.figure(
        figsize=(
            10 if len(class_names) <= 10 else 16,
            8 if len(class_names) <= 10 else 14,
        )
    )
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        linewidths=0.5,
        linecolor="lightgray",
    )
    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Predicted Class", fontsize=12, labelpad=10)
    plt.ylabel("True Class", fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_binary_roc_pr(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[Path] = None,
) -> None:
    """
    Generates side-by-side ROC and Precision-Recall curves for binary classification.
    """
    prob_pos = y_prob[:, 1] if y_prob.ndim == 2 else y_prob

    fpr, tpr, _ = roc_curve(y_true, prob_pos)
    roc_auc_val = auc(fpr, tpr)

    precision_pts, recall_pts, _ = precision_recall_curve(y_true, prob_pos)
    pr_auc_val = auc(recall_pts, precision_pts)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ROC Curve
    axes[0].plot(
        fpr,
        tpr,
        color="#2980b9",
        lw=2.5,
        label=f"{model_name} (AUC = {roc_auc_val:.4f})",
    )
    axes[0].plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--")
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel("False Positive Rate (FPR)", fontsize=11)
    axes[0].set_ylabel("True Positive Rate (TPR / Recall)", fontsize=11)
    axes[0].set_title(
        f"Receiver Operating Characteristic (ROC) &mdash; {model_name}",
        fontsize=12,
        fontweight="bold",
    )
    axes[0].legend(loc="lower right")
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # Precision-Recall Curve
    axes[1].plot(
        recall_pts,
        precision_pts,
        color="#27ae60",
        lw=2.5,
        label=f"{model_name} (PR-AUC = {pr_auc_val:.4f})",
    )
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel("Recall", fontsize=11)
    axes[1].set_ylabel("Precision", fontsize=11)
    axes[1].set_title(
        f"Precision-Recall Curve &mdash; {model_name}",
        fontsize=12,
        fontweight="bold",
    )
    axes[1].legend(loc="lower left")
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
