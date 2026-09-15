"""
Phase 2: Functional Category Classification (8 Classes) Training & Benchmarking Runner.
Evaluates multi-class models across: DDoS, DoS, Mirai, Recon, Spoofing, Web, BruteForce, Benign.
"""

import json
from pathlib import Path
import time
from typing import Dict, List, Optional
import joblib
import numpy as np
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
    plot_confusion_matrix,
)
from src.models import get_model
from src.preprocessing import prepare_dataset


def run_category_benchmark(
    sample_path: Path = SAMPLE_FILE,
    models_to_train: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Executes Phase 2 functional category (8-class) benchmark across all models.
    """
    if models_to_train is None:
        models_to_train = [
            "logistic_regression",
            "random_forest",
            "lightgbm",
            "xgboost",
            "pytorch_dnn",
        ]

    category_models_dir = MODELS_DIR / "category_8class"
    category_models_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = category_models_dir / "category8_scaler.joblib"

    print("=================================================================", flush=True)
    print(" PHASE 2: FUNCTIONAL CATEGORY CLASSIFICATION (8 CLASSES)", flush=True)
    print("=================================================================", flush=True)
    print(f"Loading data from: {sample_path}", flush=True)
    df = pd.read_csv(sample_path)

    print("\nPreparing 8-class train/test split and applying leak-free scaling...", flush=True)
    X_train, X_test, y_train, y_test, class_names, scaler = prepare_dataset(
        df,
        target_type="category",
        test_size=0.2,
        random_state=42,
        scale_features=True,
        save_scaler_path=scaler_path,
    )

    print(f"X_train Shape: {X_train.shape} | X_test Shape: {X_test.shape}", flush=True)
    print(f"Classes ({len(class_names)}): {class_names}", flush=True)

    print("\nTraining Set Distribution by Category:", flush=True)
    train_dist = pd.Series(y_train).value_counts().sort_index()
    for idx, count in train_dist.items():
        pct = (count / len(y_train)) * 100
        print(f"  [{idx}] {class_names[idx]:<12}: {count:>7,} ({pct:5.2f}%)", flush=True)

    results_summary = []
    per_class_summaries = {}

    for model_name in models_to_train:
        print(f"\n--> Training [{model_name}] on 8 classes...", flush=True)
        model = get_model(
            model_name=model_name,
            num_classes=8,
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
        checkpoint_path = category_models_dir / f"{model_name}.joblib"
        if model_name == "pytorch_dnn":
            torch.save(model.model.state_dict(), category_models_dir / f"{model_name}.pt")
        else:
            joblib.dump(model, checkpoint_path)

        # Save 8x8 confusion matrix plot
        cm_path = FIGURES_DIR / f"category8_cm_{model_name}.png"
        plot_confusion_matrix(
            cm=metrics["confusion_matrix"],
            class_names=class_names,
            title=f"8-Class Confusion Matrix &mdash; {model_name}",
            save_path=cm_path,
            normalize=True,
            cmap="Purples",
        )

        row = {
            "Model": model_name,
            "Accuracy": metrics["accuracy"],
            "Macro F1": metrics["f1_macro"],
            "Weighted F1": metrics["f1_weighted"],
            "Macro Precision": metrics["precision_macro"],
            "Macro Recall": metrics["recall_macro"],
            "ROC-AUC (OVR)": metrics.get("roc_auc"),
            "Train Time (s)": round(train_duration, 2),
            "Latency (us/flow)": round(latency_us, 2),
            "Throughput (flows/s)": int(throughput_fps),
        }
        results_summary.append(row)
        per_class_summaries[model_name] = metrics["per_class_report"]

        print(
            f"    Done! Acc: {row['Accuracy']:.4f} | Macro F1: {row['Macro F1']:.4f} | "
            f"Latency: {row['Latency (us/flow)']} us/flow | Train Time: {row['Train Time (s)']}s",
            flush=True,
        )

    # Generate benchmark DataFrame
    benchmark_df = pd.DataFrame(results_summary)
    benchmark_df.sort_values(by="Macro F1", ascending=False, inplace=True)

    # Save to results/tables/
    csv_out = TABLES_DIR / "category8_models_benchmark.csv"
    json_out = TABLES_DIR / "category8_models_benchmark.json"
    per_class_json = TABLES_DIR / "category8_per_class_reports.json"

    benchmark_df.to_csv(csv_out, index=False)
    with open(json_out, "w") as f:
        json.dump(results_summary, f, indent=2)
    with open(per_class_json, "w") as f:
        json.dump(per_class_summaries, f, indent=2)

    print("\n=================================================================", flush=True)
    print(" PHASE 2: 8-CLASS BENCHMARK SUMMARY", flush=True)
    print("=================================================================", flush=True)
    print(tabulate(benchmark_df, headers="keys", tablefmt="pipe", showindex=False), flush=True)
    print(f"\nSaved results to: {csv_out}", flush=True)
    print(f"Saved 8x8 confusion matrices to: {FIGURES_DIR}", flush=True)

    return benchmark_df


if __name__ == "__main__":
    run_category_benchmark()
