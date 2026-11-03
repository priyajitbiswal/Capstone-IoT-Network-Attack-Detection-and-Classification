"""
Stage 3: Baseline Model Training, Validation, and Benchmarking Runner.
Executes models sequentially, measures psutil memory, evaluates strictly on validation set,
benchmarks inference latency/throughput, and logs to centralized ledger.

ZERO TEST TOUCH: Neither Internal Test nor Cross-Capture Test is ever loaded or evaluated here.
"""

import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import psutil
import joblib
import numpy as np
import pandas as pd

from src_v2.config import (
    BINARY_CLASSES,
    EIGHT_CLASSES,
    MODELS_V2_DIR,
    PARTITION_ASSIGNMENTS_FILE,
    RANDOM_SEED,
    RESULTS_V2_DIR,
    TABLES_V2_DIR,
    THIRTY_FOUR_CLASSES,
)
from src_v2.preprocessing import load_partitioned_datasets
from src_v2.models import get_model_v2
from src_v2.evaluate import (
    evaluate_predictions_v2,
    benchmark_single_flow_latency,
    benchmark_batch_throughput,
)

METRICS_V2_DIR = RESULTS_V2_DIR / "metrics"
PER_CLASS_DIR = METRICS_V2_DIR / "per_class"
LEDGER_CSV = TABLES_V2_DIR / "centralized_experiment_ledger.csv"

for p in [METRICS_V2_DIR, PER_CLASS_DIR, TABLES_V2_DIR, MODELS_V2_DIR]:
    p.mkdir(parents=True, exist_ok=True)


def get_process_memory_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def run_single_experiment(
    exp_id: str,
    target_type: str,
    model_name: str,
    assignments_file: Path = PARTITION_ASSIGNMENTS_FILE,
    strategy_name: str = "Strategy 1 (Primary)",
    model_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Executes a single model training and validation pipeline:
    1. Loads train and val strictly.
    2. Measures RAM and training duration.
    3. Evaluates predictions on validation set.
    4. Benchmarks single-flow latency and batch throughput.
    5. Saves model artifact and logs results.
    """
    model_params = model_params or {}
    print(f"\n{'='*70}")
    print(f"RUNNING EXPERIMENT: {exp_id}")
    print(f"Task: {target_type} | Model: {model_name} | Strategy: {strategy_name}")
    print(f"{'='*70}")

    # Track initial memory
    gc.collect()
    rss_start = get_process_memory_mb()

    # 1. Load Data (fit on train, transform val)
    print("[1/5] Loading partitioned datasets (Train & Val only)...")
    (X_train, y_train), (X_val, y_val), _, class_names, preprocessor = load_partitioned_datasets(
        target_type=target_type,
        scale_features=True,
        assignments_file=assignments_file,
    )
    num_classes = len(class_names)
    print(f"      Train shape: {X_train.shape}, Val shape: {X_val.shape}, Classes: {num_classes}")

    # 2. Initialize Model
    print(f"[2/5] Initializing {model_name}...")
    model = get_model_v2(
        model_name=model_name,
        num_classes=num_classes,
        input_dim=X_train.shape[1],
        random_state=RANDOM_SEED,
        **model_params,
    )

    # 3. Train Model & Measure Resource Consumption
    print("[3/5] Fitting model strictly on training data...")
    t_start = time.perf_counter()
    rss_peak_tracker = [get_process_memory_mb()]

    if model_name == "pytorch_dnn":
        # Pass validation data strictly for loss recording
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    else:
        model.fit(X_train, y_train)

    train_duration = time.perf_counter() - t_start
    rss_after_fit = get_process_memory_mb()
    rss_peak_mb = max(rss_peak_tracker + [rss_after_fit])
    print(f"      Fit completed in {train_duration:.2f}s | Peak RSS: {rss_peak_mb:.1f} MB (Delta: {rss_peak_mb - rss_start:+.1f} MB)")

    # 4. Evaluate on Validation Set
    print("[4/5] Evaluating predictions on Validation partition...")
    t_val_start = time.perf_counter()
    y_pred = model.predict(X_val)
    val_predict_duration = time.perf_counter() - t_val_start

    y_prob = None
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_val)
        except Exception as e:
            print(f"      Warning: predict_proba failed: {e}")

    metrics = evaluate_predictions_v2(
        y_true=y_val,
        y_pred=y_pred,
        y_prob=y_prob,
        class_names=class_names,
    )
    print(f"      Accuracy: {metrics['accuracy']:.4f} | Macro F1: {metrics['macro_f1']:.4f} | Weighted F1: {metrics['weighted_f1']:.4f}")
    if metrics["roc_auc_ovr"] is not None:
        print(f"      Macro OVR ROC-AUC: {metrics['roc_auc_ovr']:.4f}")
    else:
        print(f"      Macro OVR ROC-AUC: N/A ({metrics['roc_auc_error']})")

    # 5. Benchmarking Inference Latency and Throughput
    print("[5/5] Running inference latency and throughput benchmarks...")
    single_bench = benchmark_single_flow_latency(
        model=model,
        X_sample=X_val,
        n_samples=2000,
        n_warmup=500,
        n_runs=5,
        scaler=None,  # X_val is already scaled
    )
    batch_bench = benchmark_batch_throughput(
        model=model,
        X_sample=X_val,
        batch_size=10000,
        n_warmup=3,
        n_runs=5,
        scaler=None,
    )
    print(f"      Single-Flow Latency: P50={single_bench['median_latency_us']:.1f} us | P95={single_bench['p95_latency_us']:.1f} us | P99={single_bench['p99_latency_us']:.1f} us")
    print(f"      Batch Throughput:    {batch_bench['throughput_predictions_per_sec']:,.0f} samples/sec | Amortized: {batch_bench['amortized_latency_us']:.2f} us/sample")

    # Save Model Artifact
    model_save_path = MODELS_V2_DIR / f"{exp_id}.joblib"
    if model_name == "pytorch_dnn":
        # Save state dict and wrapper metadata
        torch_save_path = MODELS_V2_DIR / f"{exp_id}_torch.pt"
        import torch
        torch.save(model.model.state_dict(), torch_save_path)
        # Also save DNN loss history
        dnn_loss_path = METRICS_V2_DIR / f"{exp_id}_dnn_loss.json"
        with open(dnn_loss_path, "w") as f:
            json.dump(model.history, f, indent=2)
        print(f"      Saved PyTorch state dict to {torch_save_path}")
        print(f"      Saved DNN loss history to {dnn_loss_path}")
    else:
        joblib.dump(model, model_save_path)
        print(f"      Saved model checkpoint to {model_save_path}")

    # Save Per-Class Classification Report
    per_class_path = PER_CLASS_DIR / f"{exp_id}_per_class.json"
    with open(per_class_path, "w") as f:
        json.dump(metrics["per_class_report"], f, indent=2)

    # Compile Experiment Record for Centralized Ledger
    ledger_record = {
        "experiment_id": exp_id,
        "target_type": target_type,
        "model_name": model_name,
        "strategy": strategy_name,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "num_classes": num_classes,
        "accuracy": round(metrics["accuracy"], 5),
        "macro_f1": round(metrics["macro_f1"], 5),
        "weighted_f1": round(metrics["weighted_f1"], 5),
        "macro_precision": round(metrics["macro_precision"], 5),
        "macro_recall": round(metrics["macro_recall"], 5),
        "roc_auc_ovr": round(metrics["roc_auc_ovr"], 5) if metrics["roc_auc_ovr"] is not None else None,
        "roc_auc_status": "Calculated" if metrics["roc_auc_ovr"] is not None else metrics["roc_auc_error"],
        "train_duration_sec": round(train_duration, 2),
        "val_predict_sec": round(val_predict_duration, 3),
        "peak_rss_mb": round(rss_peak_mb, 1),
        "rss_delta_mb": round(rss_peak_mb - rss_start, 1),
        "single_latency_p50_us": single_bench["median_latency_us"],
        "single_latency_p95_us": single_bench["p95_latency_us"],
        "single_latency_p99_us": single_bench["p99_latency_us"],
        "batch_throughput_samples_per_sec": batch_bench["throughput_predictions_per_sec"],
        "batch_amortized_us": batch_bench["amortized_latency_us"],
        "model_params": json.dumps(model_params),
        "random_seed": RANDOM_SEED,
    }

    # Append to Centralized Ledger CSV
    df_new = pd.DataFrame([ledger_record])
    if LEDGER_CSV.exists():
        df_existing = pd.read_csv(LEDGER_CSV)
        # Update or append
        df_combined = pd.concat([df_existing[df_existing["experiment_id"] != exp_id], df_new], ignore_index=True)
        df_combined.to_csv(LEDGER_CSV, index=False)
    else:
        df_new.to_csv(LEDGER_CSV, index=False)

    print(f"      Recorded results to: {LEDGER_CSV}")

    # Cleanup memory
    del model, X_train, y_train, X_val, y_val
    gc.collect()

    return ledger_record


# Defined list of all Stage 3 Experiments
EXPERIMENT_REGISTRY: List[Dict[str, Any]] = [
    # --- PHASE 1: BINARY CLASSIFICATION (5 Models) ---
    {
        "exp_id": "binary_logistic_regression",
        "target_type": "binary",
        "model_name": "logistic_regression",
        "strategy_name": "Strategy 1 (Primary)",
        "assignments_file": PARTITION_ASSIGNMENTS_FILE,
        "model_params": {"max_iter": 100},
    },
    {
        "exp_id": "binary_random_forest",
        "target_type": "binary",
        "model_name": "random_forest",
        "strategy_name": "Strategy 1 (Primary)",
        "assignments_file": PARTITION_ASSIGNMENTS_FILE,
        "model_params": {"n_estimators": 50, "max_depth": 15},
    },
    {
        "exp_id": "binary_lightgbm",
        "target_type": "binary",
        "model_name": "lightgbm",
        "strategy_name": "Strategy 1 (Primary)",
        "assignments_file": PARTITION_ASSIGNMENTS_FILE,
        "model_params": {"n_estimators": 100, "learning_rate": 0.05, "num_leaves": 31},
    },
    {
        "exp_id": "binary_xgboost",
        "target_type": "binary",
        "model_name": "xgboost",
        "strategy_name": "Strategy 1 (Primary)",
        "assignments_file": PARTITION_ASSIGNMENTS_FILE,
        "model_params": {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 6},
    },
    {
        "exp_id": "binary_pytorch_dnn",
        "target_type": "binary",
        "model_name": "pytorch_dnn",
        "strategy_name": "Strategy 1 (Primary)",
        "assignments_file": PARTITION_ASSIGNMENTS_FILE,
        "model_params": {"epochs": 5, "batch_size": 2048, "lr": 0.002},
    },

    # --- PHASE 2: 8-CLASS CATEGORY CLASSIFICATION (5 Models) ---
    {
        "exp_id": "category_logistic_regression",
        "target_type": "category",
        "model_name": "logistic_regression",
        "strategy_name": "Strategy 1 (Primary)",
        "assignments_file": PARTITION_ASSIGNMENTS_FILE,
        "model_params": {"max_iter": 100},
    },
    {
        "exp_id": "category_random_forest",
        "target_type": "category",
        "model_name": "random_forest",
        "strategy_name": "Strategy 1 (Primary)",
        "assignments_file": PARTITION_ASSIGNMENTS_FILE,
        "model_params": {"n_estimators": 50, "max_depth": 15},
    },
    {
        "exp_id": "category_lightgbm",
        "target_type": "category",
        "model_name": "lightgbm",
        "strategy_name": "Strategy 1 (Primary)",
        "assignments_file": PARTITION_ASSIGNMENTS_FILE,
        "model_params": {"n_estimators": 100, "learning_rate": 0.05, "num_leaves": 31},
    },
    {
        "exp_id": "category_xgboost",
        "target_type": "category",
        "model_name": "xgboost",
        "strategy_name": "Strategy 1 (Primary)",
        "assignments_file": PARTITION_ASSIGNMENTS_FILE,
        "model_params": {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 6},
    },
    {
        "exp_id": "category_pytorch_dnn",
        "target_type": "category",
        "model_name": "pytorch_dnn",
        "strategy_name": "Strategy 1 (Primary)",
        "assignments_file": PARTITION_ASSIGNMENTS_FILE,
        "model_params": {"epochs": 5, "batch_size": 2048, "lr": 0.002},
    },

    # --- PHASE 3: 34-CLASS FINE-GRAINED CLASSIFICATION (5 Models) ---
    {
        "exp_id": "fine_grained_logistic_regression",
        "target_type": "fine_grained",
        "model_name": "logistic_regression",
        "strategy_name": "Strategy 1 (Primary)",
        "assignments_file": PARTITION_ASSIGNMENTS_FILE,
        "model_params": {"max_iter": 100},
    },
    {
        "exp_id": "fine_grained_random_forest",
        "target_type": "fine_grained",
        "model_name": "random_forest",
        "strategy_name": "Strategy 1 (Primary)",
        "assignments_file": PARTITION_ASSIGNMENTS_FILE,
        "model_params": {"n_estimators": 50, "max_depth": 15},
    },
    {
        "exp_id": "fine_grained_lightgbm",
        "target_type": "fine_grained",
        "model_name": "lightgbm",
        "strategy_name": "Strategy 1 (Primary)",
        "assignments_file": PARTITION_ASSIGNMENTS_FILE,
        "model_params": {"n_estimators": 100, "learning_rate": 0.05, "num_leaves": 31},
    },
    {
        "exp_id": "fine_grained_xgboost",
        "target_type": "fine_grained",
        "model_name": "xgboost",
        "strategy_name": "Strategy 1 (Primary)",
        "assignments_file": PARTITION_ASSIGNMENTS_FILE,
        "model_params": {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 6},
    },
    {
        "exp_id": "fine_grained_pytorch_dnn",
        "target_type": "fine_grained",
        "model_name": "pytorch_dnn",
        "strategy_name": "Strategy 1 (Primary)",
        "assignments_file": PARTITION_ASSIGNMENTS_FILE,
        "model_params": {"epochs": 5, "batch_size": 2048, "lr": 0.002},
    },

    # --- PHASE 4: STRATEGY 2 ABLATION (Ambiguous Vectors Purged) ---
    {
        "exp_id": "ablation_s2_binary_lightgbm",
        "target_type": "binary",
        "model_name": "lightgbm",
        "strategy_name": "Strategy 2 (Ambiguous Purged)",
        "assignments_file": PARTITION_ASSIGNMENTS_FILE.parent / "ablation_strategy2" / "partition_assignments.csv",
        "model_params": {"n_estimators": 100, "learning_rate": 0.05, "num_leaves": 31},
    },
    {
        "exp_id": "ablation_s2_category_lightgbm",
        "target_type": "category",
        "model_name": "lightgbm",
        "strategy_name": "Strategy 2 (Ambiguous Purged)",
        "assignments_file": PARTITION_ASSIGNMENTS_FILE.parent / "ablation_strategy2" / "partition_assignments.csv",
        "model_params": {"n_estimators": 100, "learning_rate": 0.05, "num_leaves": 31},
    },
    {
        "exp_id": "ablation_s2_fine_grained_lightgbm",
        "target_type": "fine_grained",
        "model_name": "lightgbm",
        "strategy_name": "Strategy 2 (Ambiguous Purged)",
        "assignments_file": PARTITION_ASSIGNMENTS_FILE.parent / "ablation_strategy2" / "partition_assignments.csv",
        "model_params": {"n_estimators": 100, "learning_rate": 0.05, "num_leaves": 31},
    },
    {
        "exp_id": "ablation_s2_fine_grained_xgboost",
        "target_type": "fine_grained",
        "model_name": "xgboost",
        "strategy_name": "Strategy 2 (Ambiguous Purged)",
        "assignments_file": PARTITION_ASSIGNMENTS_FILE.parent / "ablation_strategy2" / "partition_assignments.csv",
        "model_params": {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 6},
    },
]


def print_experiment_plan():
    print("=" * 80)
    print("STAGE 3 EXPERIMENT SUITE CATALOG (Sequential Execution Plan)")
    print("=" * 80)
    for idx, exp in enumerate(EXPERIMENT_REGISTRY, start=1):
        print(f"[{idx:02d}/19] ID: {exp['exp_id']:<35} | Task: {exp['target_type']:<12} | Strategy: {exp['strategy_name']}")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stage 3 Model Training and Benchmark Runner")
    parser.add_argument("--list", action="store_true", help="List all planned experiments")
    parser.add_argument("--exp", type=str, default=None, help="Run a specific experiment by ID")
    parser.add_argument("--phase", type=str, default=None, choices=["binary", "category", "fine_grained", "ablation"], help="Run all experiments in a specific phase")
    parser.add_argument("--all", action="store_true", help="Run all experiments sequentially")
    args = parser.parse_args()

    if args.list or (len(sys.argv) == 1):
        print_experiment_plan()
    elif args.exp:
        matched = [e for e in EXPERIMENT_REGISTRY if e["exp_id"] == args.exp]
        if not matched:
            print(f"Error: Experiment ID '{args.exp}' not found in registry.")
            sys.exit(1)
        run_single_experiment(**matched[0])
    elif args.phase:
        if args.phase == "ablation":
            targets = [e for e in EXPERIMENT_REGISTRY if "ablation" in e["exp_id"]]
        else:
            targets = [e for e in EXPERIMENT_REGISTRY if e["target_type"] == args.phase and "ablation" not in e["exp_id"]]
        print(f"Running phase '{args.phase}' ({len(targets)} experiments sequentially)...")
        for t in targets:
            run_single_experiment(**t)
    elif args.all:
        print(f"Running all {len(EXPERIMENT_REGISTRY)} experiments sequentially...")
        for t in EXPERIMENT_REGISTRY:
            run_single_experiment(**t)

