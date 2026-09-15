"""
Step 5: Feature Importance Analysis & Inference Latency Optimization.
Computes multi-model feature importances (RF Gini, XGB Gain, LGBM Split),
prunes redundant features, evaluates accuracy delta vs latency speedup,
and saves publication-ready figures and benchmarks.
"""

import json
from pathlib import Path
import time
from typing import Dict, List, Tuple
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tabulate import tabulate

from src.config import (
    FEATURE_COLUMNS,
    FIGURES_DIR,
    MODELS_DIR,
    SAMPLE_FILE,
    TABLES_DIR,
)
from src.evaluate import evaluate_predictions, measure_inference_latency
from src.models import get_model
from src.preprocessing import clean_features, encode_labels, prepare_dataset


def compute_feature_importances(
    models_dir: Path = MODELS_DIR / "attack_34class",
) -> pd.DataFrame:
    """
    Extracts and normalizes feature importance scores from trained tree models.
    Computes ensemble consensus importance ranking.
    """
    rf = joblib.load(models_dir / "random_forest.joblib")
    xgb = joblib.load(models_dir / "xgboost.joblib")
    lgb = joblib.load(models_dir / "lightgbm.joblib")

    rf_imp = rf.feature_importances_
    xgb_imp = xgb.feature_importances_
    lgb_raw = lgb.feature_importances_
    lgb_imp = lgb_raw / lgb_raw.sum() if lgb_raw.sum() > 0 else lgb_raw

    # Normalize RF and XGB to sum to 1.0 for fair weighting
    rf_norm = rf_imp / rf_imp.sum()
    xgb_norm = xgb_imp / xgb_imp.sum()

    df_imp = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "RF_Gini": rf_norm,
        "XGB_Gain": xgb_norm,
        "LGBM_Split": lgb_imp,
    })
    df_imp["Consensus_Score"] = (df_imp["RF_Gini"] + df_imp["XGB_Gain"] + df_imp["LGBM_Split"]) / 3.0
    df_imp.sort_values(by="Consensus_Score", ascending=False, inplace=True)
    df_imp["Cumulative_Score"] = df_imp["Consensus_Score"].cumsum()
    df_imp["Rank"] = range(1, len(df_imp) + 1)

    # Save to CSV and JSON
    csv_path = TABLES_DIR / "feature_importance_rankings.csv"
    df_imp.to_csv(csv_path, index=False)
    print(f"Saved feature importance rankings to: {csv_path}", flush=True)

    return df_imp


def plot_feature_importance(df_imp: pd.DataFrame, top_k: int = 20) -> None:
    """
    Plots horizontal bar chart of top K features and cumulative score curve.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9), gridspec_kw={"width_ratios": [2, 1]})

    top_df = df_imp.head(top_k).copy()
    top_df.sort_values(by="Consensus_Score", ascending=True, inplace=True)

    # Horizontal bar plot
    colors = sns.color_palette("Blues_d", n_colors=top_k)
    bars = ax1.barh(top_df["Feature"], top_df["Consensus_Score"] * 100, color=colors)
    ax1.set_title(f"Top {top_k} Features by Consensus Importance Score", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Consensus Importance (%)", fontsize=11)
    ax1.set_ylabel("Feature Name", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6, axis="x")

    # Add data labels
    for bar in bars:
        w = bar.get_width()
        ax1.text(w + 0.1, bar.get_y() + bar.get_height() / 2, f"{w:.2f}%", va="center", fontsize=9)

    # Cumulative Importance Curve across all 39 features
    ax2.plot(range(1, len(df_imp) + 1), df_imp["Cumulative_Score"] * 100, marker="o", color="#2980b9", lw=2.5)
    ax2.axvline(x=top_k, color="#e74c3c", linestyle="--", lw=2, label=f"Top {top_k} Cutoff ({df_imp.iloc[top_k-1]['Cumulative_Score']*100:.1f}%)")
    ax2.axhline(y=df_imp.iloc[top_k-1]["Cumulative_Score"] * 100, color="#e74c3c", linestyle=":", lw=1.5)
    ax2.set_title("Cumulative Information Retained", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Number of Features", fontsize=11)
    ax2.set_ylabel("Cumulative Score (%)", fontsize=11)
    ax2.set_ylim([0, 105])
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right")

    plt.tight_layout()
    plot_path = FIGURES_DIR / "feature_importance_ranking.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved feature importance figure to: {plot_path}", flush=True)


def benchmark_feature_reduction(
    top_k: int = 20,
    target_type: str = "fine_grained",
    sample_path: Path = SAMPLE_FILE,
) -> pd.DataFrame:
    """
    Compares model performance and inference latency between Full 39 Features
    and Reduced top-K features.
    """
    df_imp = compute_feature_importances()
    top_features = df_imp.head(top_k)["Feature"].tolist()

    reduced_models_dir = MODELS_DIR / "reduced_features"
    reduced_models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(top_features, reduced_models_dir / f"top_{top_k}_feature_names.joblib")

    print(f"\n=================================================================", flush=True)
    print(f" FEATURE SPACE REDUCTION BENCHMARK (39 vs {top_k} Features)", flush=True)
    print(f" Selected Top {top_k} Features: {top_features}", flush=True)
    print(f"=================================================================", flush=True)

    df = pd.read_csv(sample_path)
    y, _, class_names = encode_labels(df, target_type=target_type)
    num_classes = len(class_names)

    # 1. Full 39 Features
    X_full = clean_features(df, FEATURE_COLUMNS).values
    # 2. Reduced Features
    X_red = clean_features(df, top_features).values

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # Split identically
    X_tr_full, X_te_full, y_train, y_test = train_test_split(
        X_full, y, test_size=0.2, random_state=42, stratify=y
    )
    X_tr_red, X_te_red, _, _ = train_test_split(
        X_red, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler_full = StandardScaler()
    X_tr_full = scaler_full.fit_transform(X_tr_full).astype(np.float32)
    X_te_full = scaler_full.transform(X_te_full).astype(np.float32)

    scaler_red = StandardScaler()
    X_tr_red = scaler_red.fit_transform(X_tr_red).astype(np.float32)
    X_te_red = scaler_red.transform(X_te_red).astype(np.float32)
    joblib.dump(scaler_red, reduced_models_dir / f"reduced_{top_k}_scaler.joblib")

    models_to_test = ["random_forest", "lightgbm", "xgboost"]
    model_kwargs = {
        "random_forest": {"n_estimators": 50, "max_depth": 15},
        "lightgbm": {"n_estimators": 80, "learning_rate": 0.08, "num_leaves": 31},
        "xgboost": {"n_estimators": 60, "learning_rate": 0.08, "max_depth": 6},
    }

    comparison_rows = []

    for m_name in models_to_test:
        kwargs = model_kwargs.get(m_name, {})
        print(f"\nEvaluating Model: [{m_name}]", flush=True)

        # A) Full Features
        print(f"  --> Training on Full 39 Features...", flush=True)
        m_full = get_model(m_name, num_classes=num_classes, input_dim=39, class_weight="balanced", random_state=42, **kwargs)
        t0 = time.time()
        m_full.fit(X_tr_full, y_train)
        time_full = time.time() - t0
        y_pred_full = m_full.predict(X_te_full)
        met_full = evaluate_predictions(y_test, y_pred_full, class_names=class_names)
        lat_full, fps_full = measure_inference_latency(m_full, X_te_full, n_runs=5, n_instances=10_000)

        # B) Reduced Features
        print(f"  --> Training on Reduced {top_k} Features...", flush=True)
        m_red = get_model(m_name, num_classes=num_classes, input_dim=top_k, class_weight="balanced", random_state=42, **kwargs)
        t0 = time.time()
        m_red.fit(X_tr_red, y_train)
        time_red = time.time() - t0
        y_pred_red = m_red.predict(X_te_red)
        met_red = evaluate_predictions(y_test, y_pred_red, class_names=class_names)
        lat_red, fps_red = measure_inference_latency(m_red, X_te_red, n_runs=5, n_instances=10_000)

        joblib.dump(m_red, reduced_models_dir / f"{m_name}_top{top_k}.joblib")

        # Deltas
        acc_delta = (met_red["accuracy"] - met_full["accuracy"]) * 100
        f1_delta = (met_red["f1_macro"] - met_full["f1_macro"]) * 100
        speedup = ((lat_full - lat_red) / lat_full) * 100 if lat_full > 0 else 0.0

        comparison_rows.append({
            "Model": m_name,
            "Feature Set": "Full (39 feats)",
            "Features Count": 39,
            "Accuracy (%)": round(met_full["accuracy"] * 100, 2),
            "Macro F1": round(met_full["f1_macro"], 4),
            "Weighted F1": round(met_full["f1_weighted"], 4),
            "Train Time (s)": round(time_full, 2),
            "Latency (us/flow)": round(lat_full, 2),
            "Throughput (fps)": int(fps_full),
            "Acc Delta (%)": 0.0,
            "Speedup (%)": 0.0,
        })
        comparison_rows.append({
            "Model": m_name,
            "Feature Set": f"Reduced ({top_k} feats)",
            "Features Count": top_k,
            "Accuracy (%)": round(met_red["accuracy"] * 100, 2),
            "Macro F1": round(met_red["f1_macro"], 4),
            "Weighted F1": round(met_red["f1_weighted"], 4),
            "Train Time (s)": round(time_red, 2),
            "Latency (us/flow)": round(lat_red, 2),
            "Throughput (fps)": int(fps_red),
            "Acc Delta (%)": round(acc_delta, 2),
            "Speedup (%)": round(speedup, 2),
        })

    comp_df = pd.DataFrame(comparison_rows)
    csv_out = TABLES_DIR / "feature_reduction_benchmark.csv"
    json_out = TABLES_DIR / "feature_reduction_benchmark.json"
    comp_df.to_csv(csv_out, index=False)
    with open(json_out, "w") as f:
        json.dump(comparison_rows, f, indent=2)

    print("\n=================================================================", flush=True)
    print(" FEATURE REDUCTION BENCHMARK RESULTS", flush=True)
    print("=================================================================", flush=True)
    print(tabulate(comp_df, headers="keys", tablefmt="pipe", showindex=False), flush=True)
    print(f"\nSaved benchmark table to: {csv_out}", flush=True)

    # Plot Trade-off Comparison
    plot_reduction_tradeoffs(comp_df, top_k)

    return comp_df


def plot_reduction_tradeoffs(comp_df: pd.DataFrame, top_k: int) -> None:
    """
    Generates comparison plots showing accuracy retention and latency speedup.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Accuracy Retention
    sns.barplot(data=comp_df, x="Model", y="Accuracy (%)", hue="Feature Set", ax=axes[0], palette="Blues_r")
    axes[0].set_title("Classification Accuracy Retention", fontsize=12, fontweight="bold")
    axes[0].set_ylim([60, 80])
    axes[0].set_ylabel("Accuracy (%)", fontsize=11)
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # 2. Macro F1 Retention
    sns.barplot(data=comp_df, x="Model", y="Macro F1", hue="Feature Set", ax=axes[1], palette="Greens_r")
    axes[1].set_title("Macro F1-Score Retention", fontsize=12, fontweight="bold")
    axes[1].set_ylim([0.50, 0.70])
    axes[1].set_ylabel("Macro F1", fontsize=11)
    axes[1].grid(True, linestyle=":", alpha=0.6)

    # 3. Inference Latency Reduction (Speedup)
    sns.barplot(data=comp_df, x="Model", y="Latency (us/flow)", hue="Feature Set", ax=axes[2], palette="Reds_r")
    axes[2].set_title("Per-Flow Inference Latency (us)", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("Latency (us/flow) [Lower is Better]", fontsize=11)
    axes[2].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "feature_reduction_comparison.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved reduction comparison figure to: {fig_path}", flush=True)


if __name__ == "__main__":
    df_imp = compute_feature_importances()
    plot_feature_importance(df_imp, top_k=20)
    benchmark_feature_reduction(top_k=20, target_type="fine_grained")
