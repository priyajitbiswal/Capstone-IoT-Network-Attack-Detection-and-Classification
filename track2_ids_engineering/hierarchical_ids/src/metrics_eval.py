"""
Comprehensive Evaluation, Stage Decomposition & Benchmarking Utilities for E1-E4.

Reuses Track 1's `evaluate_predictions_v2`, `benchmark_single_flow_latency`, and
`benchmark_batch_throughput` while adding minority-attack summaries, Flood vs. Non-Flood
metrics, Benign FPR, E2 stage-wise error attribution, and E3 fusion disagreement analysis.
"""

import os
from typing import Any, Dict, Optional
import numpy as np
import psutil
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from track1_dataset_evaluation.src.config import (
    EIGHT_CLASSES,
    EIGHT_TO_IDX,
    THIRTY_FOUR_CLASSES,
)
from track1_dataset_evaluation.src.evaluate import (
    benchmark_batch_throughput,
    benchmark_single_flow_latency,
    evaluate_predictions_v2,
)
from .architectures import BaseIDSArchitecture
from .data_utils import (
    BENIGN_GLOBAL_IDX,
    FLOOD_CATEGORIES,
    FLOOD_CLASSES,
    FLOOD_GLOBAL_TO_LOCAL,
    MINORITY_ATTACK_CLASSES,
    NON_FLOOD_CLASSES,
    NON_FLOOD_GLOBAL_TO_LOCAL,
)

_FLOOD_CAT8_SET = {EIGHT_TO_IDX[c] for c in FLOOD_CATEGORIES}
_FLOOD_GLOBAL_SET = set(FLOOD_GLOBAL_TO_LOCAL.keys())
_NON_FLOOD_GLOBAL_SET = set(NON_FLOOD_GLOBAL_TO_LOCAL.keys())


def get_rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def evaluate_end_to_end(
    arch: BaseIDSArchitecture,
    X: np.ndarray,
    y_fine_true: np.ndarray,
    y_cat_true: np.ndarray,
    is_attack_true: np.ndarray,
    is_flood_true: np.ndarray,
    is_non_flood_true: np.ndarray,
    run_latency_bench: bool = False,
    n_single_samples: int = 1000,
    n_single_runs: int = 3,
) -> Dict[str, Any]:
    """
    Runs end-to-end 34-class evaluation of an architecture on a given split,
    recording primary metrics, Flood/Non-Flood/Minority breakdowns, error attribution,
    and optional operational latency/throughput/memory benchmarks.
    """
    rss_before = get_rss_mb()
    trace = arch.predict_with_trace(X)
    rss_after = get_rss_mb()

    y_pred = trace["y_pred"]
    y_prob = trace["y_prob"]
    executions = trace["executions_per_flow"]

    base = evaluate_predictions_v2(
        y_true=y_fine_true,
        y_pred=y_pred,
        y_prob=y_prob,
        class_names=THIRTY_FOUR_CLASSES,
    )
    report = base["per_class_report"]

    # Group summaries matching Track 1 convention (mean over per-class metrics in group)
    flood_recalls = [report[c]["recall"] for c in FLOOD_CLASSES]
    flood_f1s = [report[c]["f1-score"] for c in FLOOD_CLASSES]
    non_flood_recalls = [report[c]["recall"] for c in NON_FLOOD_CLASSES]
    non_flood_f1s = [report[c]["f1-score"] for c in NON_FLOOD_CLASSES]
    minority_recalls = [report[c]["recall"] for c in MINORITY_ATTACK_CLASSES]
    minority_f1s = [report[c]["f1-score"] for c in MINORITY_ATTACK_CLASSES]

    benign_mask = ~is_attack_true
    pred_is_attack = y_pred != BENIGN_GLOBAL_IDX
    benign_recall = float(report["BENIGN"]["recall"])
    benign_fpr = float(np.mean(pred_is_attack[benign_mask])) if np.any(benign_mask) else 0.0
    attack_to_benign_fnr = (
        float(np.mean(~pred_is_attack[is_attack_true])) if np.any(is_attack_true) else 0.0
    )

    # Decompose overall 34-class errors into Binary, Flood/Non-Flood routing, and Within-Family errors
    total_errors = int(np.sum(y_pred != y_fine_true))
    pred_is_flood = np.isin(y_pred, list(_FLOOD_GLOBAL_SET))
    pred_is_non_flood = np.isin(y_pred, list(_NON_FLOOD_GLOBAL_SET))

    err_attack_to_benign = int(np.sum(is_attack_true & (~pred_is_attack)))
    err_benign_to_attack = int(np.sum(benign_mask & pred_is_attack))
    err_flood_to_nonflood = int(np.sum(is_flood_true & pred_is_non_flood))
    err_nonflood_to_flood = int(np.sum(is_non_flood_true & pred_is_flood))
    err_within_flood = int(np.sum(is_flood_true & pred_is_flood & (y_pred != y_fine_true)))
    err_within_nonflood = int(np.sum(is_non_flood_true & pred_is_non_flood & (y_pred != y_fine_true)))

    assert (
        err_attack_to_benign
        + err_benign_to_attack
        + err_flood_to_nonflood
        + err_nonflood_to_flood
        + err_within_flood
        + err_within_nonflood
        == total_errors
    ), "Error decomposition mismatch!"

    summary = {
        "architecture": arch.name,
        "n_samples": int(len(y_fine_true)),
        "accuracy": round(base["accuracy"], 5),
        "macro_f1": round(base["macro_f1"], 5),
        "weighted_f1": round(base["weighted_f1"], 5),
        "mcc": round(base["mcc"], 5),
        "macro_precision": round(base["macro_precision"], 5),
        "macro_recall": round(base["macro_recall"], 5),
        "macro_ap_ovr": round(base["macro_ap_ovr"], 5) if base["macro_ap_ovr"] is not None else None,
        "roc_auc_ovr": round(base["roc_auc_ovr"], 5) if base["roc_auc_ovr"] is not None else None,
        "roc_auc_error": base["roc_auc_error"],
        "flood_macro_f1": round(float(np.mean(flood_f1s)), 5),
        "flood_macro_recall": round(float(np.mean(flood_recalls)), 5),
        "non_flood_macro_f1": round(float(np.mean(non_flood_f1s)), 5),
        "non_flood_macro_recall": round(float(np.mean(non_flood_recalls)), 5),
        "minority_macro_f1": round(float(np.mean(minority_f1s)), 5),
        "minority_macro_recall": round(float(np.mean(minority_recalls)), 5),
        "benign_recall": round(benign_recall, 5),
        "benign_fpr": round(benign_fpr, 5),
        "attack_to_benign_fnr": round(attack_to_benign_fnr, 5),
        "mean_executions_per_flow": round(float(np.mean(executions)), 4),
        "model_size_mb": round(arch.model_size_bytes() / (1024 * 1024), 3),
        "peak_rss_mb": round(max(rss_before, rss_after), 1),
        "error_decomposition": {
            "total_errors": total_errors,
            "attack_to_benign_fn": err_attack_to_benign,
            "benign_to_attack_fp": err_benign_to_attack,
            "flood_routed_to_nonflood": err_flood_to_nonflood,
            "nonflood_routed_to_flood": err_nonflood_to_flood,
            "within_flood_confusion": err_within_flood,
            "within_nonflood_confusion": err_within_nonflood,
        },
        "minority_per_class": {
            c: {
                "precision": round(float(report[c]["precision"]), 5),
                "recall": round(float(report[c]["recall"]), 5),
                "f1": round(float(report[c]["f1-score"]), 5),
                "support": int(report[c]["support"]),
            }
            for c in MINORITY_ATTACK_CLASSES
        },
    }

    if run_latency_bench:
        # Use a deterministic mixed subset covering Benign, Flood, and Non-Flood flows
        single_b = benchmark_single_flow_latency(
            model=arch,
            X_sample=X,
            n_samples=n_single_samples,
            n_warmup=min(200, n_single_samples // 4),
            n_runs=n_single_runs,
            scaler=None,
        )
        batch_b = benchmark_batch_throughput(
            model=arch,
            X_sample=X,
            batch_size=min(20000, len(X)),
            n_warmup=2,
            n_runs=5,
            scaler=None,
        )
        summary["single_latency_p50_us"] = single_b["median_latency_us"]
        summary["single_latency_p95_us"] = single_b["p95_latency_us"]
        summary["single_latency_p99_us"] = single_b["p99_latency_us"]
        summary["batch_throughput_fps"] = batch_b["throughput_predictions_per_sec"]
        summary["batch_amortized_us"] = batch_b["amortized_latency_us"]

    # Stage-specific diagnostics for E2 / E3
    stage_diagnostics = None
    if "stage2_pred_cat" in trace:
        stage_diagnostics = analyze_e2_stages(
            trace=trace,
            y_fine_true=y_fine_true,
            y_cat_true=y_cat_true,
            is_attack_true=is_attack_true,
            is_flood_true=is_flood_true,
            is_non_flood_true=is_non_flood_true,
        )
    elif "gate_flood_prob" in trace:
        stage_diagnostics = analyze_e3_fusion(
            arch=arch,
            X=X,
            trace=trace,
            y_fine_true=y_fine_true,
            is_attack_true=is_attack_true,
            is_flood_true=is_flood_true,
            is_non_flood_true=is_non_flood_true,
        )

    return {
        "summary": summary,
        "per_class_report": report,
        "threat_tier_summary": base.get("threat_tier_summary", {}),
        "confusion_matrix": base["confusion_matrix"],
        "stage_diagnostics": stage_diagnostics,
    }


def analyze_e2_stages(
    trace: Dict[str, Any],
    y_fine_true: np.ndarray,
    y_cat_true: np.ndarray,
    is_attack_true: np.ndarray,
    is_flood_true: np.ndarray,
    is_non_flood_true: np.ndarray,
) -> Dict[str, Any]:
    """Computes detailed stage-by-stage performance and routing error metrics for E2."""
    s1_pred = trace["stage1_pred_attack"]
    route = trace["specialist_route"]
    s2_cat = trace["stage2_pred_cat"]
    y_pred = trace["y_pred"]

    # Stage 1 metrics on full population
    s1_acc = float(accuracy_score(is_attack_true, s1_pred))
    s1_attack_rec = float(recall_score(is_attack_true, s1_pred, pos_label=True, zero_division=0))
    s1_benign_rec = float(recall_score(is_attack_true, s1_pred, pos_label=False, zero_division=0))
    s1_benign_fpr = float(np.mean(s1_pred[~is_attack_true]))

    # Stage 2 metrics on true attack flows that passed Stage 1
    passed_true_att = is_attack_true & s1_pred
    passed_all = s1_pred

    s2_cat_acc_on_passed_attacks = (
        float(accuracy_score(y_cat_true[passed_true_att], s2_cat[passed_true_att]))
        if np.any(passed_true_att)
        else 0.0
    )

    routed_flood = route == "3A_FLOOD"
    routed_non_flood = route == "3B_NON_FLOOD"

    # Flood routing P/R evaluated on Stage-1-passed true attack flows
    flood_route_prec = (
        float(precision_score(is_flood_true[passed_true_att], routed_flood[passed_true_att], zero_division=0))
        if np.any(passed_true_att)
        else 0.0
    )
    flood_route_rec = (
        float(recall_score(is_flood_true[passed_true_att], routed_flood[passed_true_att], zero_division=0))
        if np.any(passed_true_att)
        else 0.0
    )
    # Also report end-to-end Flood routing recall out of ALL true Flood flows (including Stage 1 misses)
    flood_route_rec_e2e = float(np.sum(is_flood_true & routed_flood) / max(1, np.sum(is_flood_true)))

    # Stage 3A performance on correctly routed Flood samples
    correct_flood_mask = is_flood_true & routed_flood
    s3a_acc_correct = (
        float(accuracy_score(y_fine_true[correct_flood_mask], y_pred[correct_flood_mask]))
        if np.any(correct_flood_mask)
        else 0.0
    )
    s3a_f1_correct = (
        float(
            f1_score(
                y_fine_true[correct_flood_mask],
                y_pred[correct_flood_mask],
                labels=[THIRTY_FOUR_CLASSES.index(c) for c in FLOOD_CLASSES],
                average="macro",
                zero_division=0,
            )
        )
        if np.any(correct_flood_mask)
        else 0.0
    )

    # Stage 3B performance on correctly routed Non-Flood samples
    correct_nf_mask = is_non_flood_true & routed_non_flood
    s3b_acc_correct = (
        float(accuracy_score(y_fine_true[correct_nf_mask], y_pred[correct_nf_mask]))
        if np.any(correct_nf_mask)
        else 0.0
    )
    s3b_f1_correct = (
        float(
            f1_score(
                y_fine_true[correct_nf_mask],
                y_pred[correct_nf_mask],
                labels=[THIRTY_FOUR_CLASSES.index(c) for c in NON_FLOOD_CLASSES],
                average="macro",
                zero_division=0,
            )
        )
        if np.any(correct_nf_mask)
        else 0.0
    )

    return {
        "stage1_accuracy": round(s1_acc, 5),
        "stage1_attack_recall": round(s1_attack_rec, 5),
        "stage1_benign_recall": round(s1_benign_rec, 5),
        "stage1_benign_fpr": round(s1_benign_fpr, 5),
        "stage1_uncertain_count": int(np.sum(trace["stage1_uncertain"])),
        "stage2_cat_accuracy_on_passed_attacks": round(s2_cat_acc_on_passed_attacks, 5),
        "stage2_flood_routing_precision_on_attacks": round(flood_route_prec, 5),
        "stage2_flood_routing_recall_on_passed_attacks": round(flood_route_rec, 5),
        "stage2_flood_routing_recall_of_all_floods": round(flood_route_rec_e2e, 5),
        "stage2_invalid_benign_predictions": int(np.sum(trace["stage2_invalid_benign"])),
        "stage2_uncertain_count": int(np.sum(trace["stage2_uncertain"])),
        "stage3a_correct_route_count": int(np.sum(correct_flood_mask)),
        "stage3a_accuracy_on_correctly_routed": round(s3a_acc_correct, 5),
        "stage3a_macro_f1_on_correctly_routed": round(s3a_f1_correct, 5),
        "stage3b_correct_route_count": int(np.sum(correct_nf_mask)),
        "stage3b_accuracy_on_correctly_routed": round(s3b_acc_correct, 5),
        "stage3b_macro_f1_on_correctly_routed": round(s3b_f1_correct, 5),
        "routed_counts": {
            "benign_exit": int(np.sum(route == "BENIGN_EXIT")),
            "stage2_benign_exit": int(np.sum(route == "STAGE2_BENIGN_EXIT")),
            "stage3a_flood": int(np.sum(routed_flood)),
            "stage3b_non_flood": int(np.sum(routed_non_flood)),
            "true_flood_sent_to_3b": int(np.sum(is_flood_true & routed_non_flood)),
            "true_non_flood_sent_to_3a": int(np.sum(is_non_flood_true & routed_flood)),
            "benign_passed_to_stage2": int(np.sum((~is_attack_true) & passed_all)),
        },
    }


def analyze_e3_fusion(
    arch: Any,
    X: np.ndarray,
    trace: Dict[str, Any],
    y_fine_true: np.ndarray,
    is_attack_true: np.ndarray,
    is_flood_true: np.ndarray,
    is_non_flood_true: np.ndarray,
) -> Dict[str, Any]:
    """Computes specialist-standalone, fusion gate, and disagreement diagnostics for E3."""
    s1_pred = trace["stage1_pred_attack"]
    y_pred = trace["y_pred"]

    # Standalone Flood specialist performance on ALL true Flood samples
    p_3a_all_flood = arch.flood_model.predict_proba(X[is_flood_true])
    local_pred_3a = np.argmax(p_3a_all_flood, axis=1)
    y_true_3a_local = np.asarray([FLOOD_GLOBAL_TO_LOCAL[int(g)] for g in y_fine_true[is_flood_true]])
    s3a_standalone_acc = float(accuracy_score(y_true_3a_local, local_pred_3a))
    s3a_standalone_f1 = float(f1_score(y_true_3a_local, local_pred_3a, average="macro", zero_division=0))

    # Standalone Non-Flood specialist performance on ALL true Non-Flood samples
    p_3b_all_nf = arch.non_flood_model.predict_proba(X[is_non_flood_true])
    local_pred_3b = np.argmax(p_3b_all_nf, axis=1)
    y_true_3b_local = np.asarray([NON_FLOOD_GLOBAL_TO_LOCAL[int(g)] for g in y_fine_true[is_non_flood_true]])
    s3b_standalone_acc = float(accuracy_score(y_true_3b_local, local_pred_3b))
    s3b_standalone_f1 = float(f1_score(y_true_3b_local, local_pred_3b, average="macro", zero_division=0))

    # Fusion routing accuracy on Stage-1-passed true attack flows
    passed_true_att = is_attack_true & s1_pred
    pred_is_flood = np.isin(y_pred, list(_FLOOD_GLOBAL_SET))
    fusion_flood_prec = (
        float(precision_score(is_flood_true[passed_true_att], pred_is_flood[passed_true_att], zero_division=0))
        if np.any(passed_true_att)
        else 0.0
    )
    fusion_flood_rec = (
        float(recall_score(is_flood_true[passed_true_att], pred_is_flood[passed_true_att], zero_division=0))
        if np.any(passed_true_att)
        else 0.0
    )

    both_conf = trace["both_specialists_confident"] & s1_pred
    disagree_naive = trace["naive_vs_fused_disagree"] & s1_pred

    return {
        "stage1_accuracy": round(float(accuracy_score(is_attack_true, s1_pred)), 5),
        "stage1_attack_recall": round(float(recall_score(is_attack_true, s1_pred, pos_label=True, zero_division=0)), 5),
        "stage1_benign_recall": round(float(recall_score(is_attack_true, s1_pred, pos_label=False, zero_division=0)), 5),
        "flood_specialist_standalone_acc": round(s3a_standalone_acc, 5),
        "flood_specialist_standalone_macro_f1": round(s3a_standalone_f1, 5),
        "non_flood_specialist_standalone_acc": round(s3b_standalone_acc, 5),
        "non_flood_specialist_standalone_macro_f1": round(s3b_standalone_f1, 5),
        "fusion_flood_precision_on_passed_attacks": round(fusion_flood_prec, 5),
        "fusion_flood_recall_on_passed_attacks": round(fusion_flood_rec, 5),
        "both_specialists_confident_count": int(np.sum(both_conf)),
        "both_specialists_confident_pct_of_attacks": round(
            float(np.sum(both_conf) / max(1, np.sum(s1_pred)) * 100.0), 2
        ),
        "naive_vs_fused_family_disagreement_count": int(np.sum(disagree_naive)),
        "accuracy_when_both_confident": round(
            float(accuracy_score(y_fine_true[both_conf], y_pred[both_conf])), 5
        )
        if np.any(both_conf)
        else None,
        "true_flood_fused_as_non_flood": int(np.sum(is_flood_true & s1_pred & (~pred_is_flood))),
        "true_non_flood_fused_as_flood": int(np.sum(is_non_flood_true & s1_pred & pred_is_flood)),
    }
