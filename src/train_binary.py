"""
Phase 1: Binary Classification Training & Benchmarking Runner.
Trains, evaluates, profiles inference latency, and compares all model families on Attack vs. Benign.
"""

import json
from pathlib import Path
import time
from typing import List, Optional
import joblib
import pandas as pd
from tabulate import tabulate
import torch

from src.config import (
    FIGURES_DIR,
    MODELS_DIR,
    SAMPLE_FILE,
    TABLES_DIR,
)
from src.evaluate import (
    evaluate_predictions,
    measure_inference_latency,
    plot_binary_roc_pr,
    plot_confusion_matrix,
)
from src.models import get_model
from src.preprocessing import prepare_dataset


def run_binary_benchmark(
    sample_path: Path = SAMPLE_FILE,
    models_to_train: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Executes Phase 1 binary intrusion detection benchmark across all models.
    """
    if models_to_train is None:
        models_to_train = [
            "logistic_regression",
            "random_forest",
            "lightgbm",
            "xgboost",
            "pytorch_dnn",
        ]

    binary_models_dir = MODELS_DIR / "binary"
    binary_models_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = binary_models_dir / "binary_scaler.joblib"

    print("=================================================================", flush=True)
    print(" PHASE 1: BINARY CLASSIFICATION BENCHMARK (2 CLASSES)", flush=True)
    print("=================================================================", flush=True)
    print(f"Loading data from: {sample_path}", flush=True)
    df = pd.read_csv(sample_path)

    print("\nPreparing train/test split and applying leak-free scaling...", flush=True)
    X_train, X_test, y_train, y_test, class_names, scaler = prepare_dataset(
        df,
        target_type="binary",
        test_size=0.2,
        random_state=42,
        scale_features=True,
        save_scaler_path=scaler_path,
    )

    print(f"X_train Shape: {X_train.shape} | X_test Shape: {X_test.shape}", flush=True)
    print(f"Train Benign: {(y_train == 0).sum():,} | Train Attack: {(y_train == 1).sum():,}", flush=True)
    print(f"Test Benign : {(y_test == 0).sum():,}  | Test Attack : {(y_test == 1).sum():,}\n", flush=True)

    results_summary = []

    for model_name in models_to_train:
        print(f"--> Training [{model_name}]...", flush=True)
        model = get_model(
            model_name=model_name,
            num_classes=2,
            input_dim=X_train.shape[1],
            class_weight="balanced",
            random_state=42,
        )

        # Train with duration profiling
        t0 = time.time()
        model.fit(X_train, y_train)
        train_duration = time.time() - t0

        # Predict
        y_pred = model.predict(X_test)
        y_prob = None
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)

        # Evaluate metrics
        metrics = evaluate_predictions(
            y_true=y_test,
            y_pred=y_pred,
            y_prob=y_prob,
            class_names=class_names,
        )

        # Measure latency & throughput
        latency_us, throughput_fps = measure_inference_latency(
            model=model,
            X_sample=X_test,
            n_runs=5,
            n_instances=10_000,
        )

        # Save model checkpoint
        checkpoint_path = binary_models_dir / f"{model_name}.joblib"
        if model_name == "pytorch_dnn":
            torch.save(model.model.state_dict(), binary_models_dir / f"{model_name}.pt")
        else:
            joblib.dump(model, checkpoint_path)

        # Save confusion matrix plot
        cm_path = FIGURES_DIR / f"binary_cm_{model_name}.png"
        plot_confusion_matrix(
            cm=metrics["confusion_matrix"],
            class_names=class_names,
            title=f"Binary Confusion Matrix &mdash; {model_name}",
            save_path=cm_path,
            normalize=True,
        )

        # Save ROC/PR curves
        if y_prob is not None:
            roc_pr_path = FIGURES_DIR / f"binary_roc_pr_{model_name}.png"
            plot_binary_roc_pr(
                y_true=y_test,
                y_prob=y_prob,
                model_name=model_name,
                save_path=roc_pr_path,
            )

        row = {
            "Model": model_name,
            "Accuracy": metrics["accuracy"],
            "Macro F1": metrics["f1_macro"],
            "Weighted F1": metrics["f1_weighted"],
            "Macro Precision": metrics["precision_macro"],
            "Macro Recall": metrics["recall_macro"],
            "ROC-AUC": metrics.get("roc_auc"),
            "PR-AUC": metrics.get("pr_auc"),
            "Train Time (s)": round(train_duration, 2),
            "Latency (us/flow)": round(latency_us, 2),
            "Throughput (flows/s)": int(throughput_fps),
        }
        results_summary.append(row)
        print(f"    Done! Acc: {row['Accuracy']:.4f} | Macro F1: {row['Macro F1']:.4f} | Latency: {row['Latency (us/flow)']} us/flow", flush=True)

    # Generate benchmark DataFrame
    benchmark_df = pd.DataFrame(results_summary)
    benchmark_df.sort_values(by="Macro F1", ascending=False, inplace=True)

    # Save to results/tables/
    csv_out = TABLES_DIR / "binary_models_benchmark.csv"
    json_out = TABLES_DIR / "binary_models_benchmark.json"
    benchmark_df.to_csv(csv_out, index=False)
    with open(json_out, "w") as f:
        json.dump(results_summary, f, indent=2)

    print("\n=================================================================", flush=True)
    print(" PHASE 1 BENCHMARK SUMMARY", flush=True)
    print("=================================================================", flush=True)
    print(tabulate(benchmark_df, headers="keys", tablefmt="pipe", showindex=False), flush=True)
    print(f"\nSaved results to: {csv_out}", flush=True)
    print(f"Saved confusion matrices and curves to: {FIGURES_DIR}", flush=True)

    return benchmark_df


if __name__ == "__main__":
    run_binary_benchmark()
