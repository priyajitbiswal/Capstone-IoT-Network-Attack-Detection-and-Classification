"""
Step 6: Master Capstone Synthesis & Comparative Presentation Engine.
Aggregates benchmarks across Binary (2-Class), Functional Category (8-Class),
Fine-Grained Attack (34-Class), and Reduced-Feature (20-Feature) tiers.
Generates master tables and publication-quality multi-panel visualization.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tabulate import tabulate

from src.config import FIGURES_DIR, TABLES_DIR

sns.set_theme(style="whitegrid", palette="muted")


def generate_master_synthesis() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Synthesizes all project benchmarks into master comparison tables and figures.
    """
    binary_csv = TABLES_DIR / "binary_models_benchmark.csv"
    category8_csv = TABLES_DIR / "category8_models_benchmark.csv"
    attack34_csv = TABLES_DIR / "attack34_models_benchmark.csv"
    reduction_csv = TABLES_DIR / "feature_reduction_benchmark.csv"

    df_bin = pd.read_csv(binary_csv)
    df_cat = pd.read_csv(category8_csv)
    df_att = pd.read_csv(attack34_csv)
    df_red = pd.read_csv(reduction_csv)

    # 1. Master Model Comparison Table Across All Granularities
    records = []

    # Map model rows from Binary
    for _, row in df_bin.iterrows():
        records.append({
            "Phase": "Phase 1: Binary",
            "Classes": 2,
            "Features": 39,
            "Model": row["Model"],
            "Accuracy (%)": round(row["Accuracy"] * 100, 2),
            "Macro F1": round(row["Macro F1"], 4),
            "Weighted F1": round(row["Weighted F1"], 4),
            "Latency (us/flow)": round(row["Latency (us/flow)"], 2),
            "Throughput (fps)": int(row["Throughput (flows/s)"]),
        })

    # Map model rows from Category 8
    for _, row in df_cat.iterrows():
        records.append({
            "Phase": "Phase 2: Category",
            "Classes": 8,
            "Features": 39,
            "Model": row["Model"],
            "Accuracy (%)": round(row["Accuracy"] * 100, 2),
            "Macro F1": round(row["Macro F1"], 4),
            "Weighted F1": round(row["Weighted F1"], 4),
            "Latency (us/flow)": round(row["Latency (us/flow)"], 2),
            "Throughput (fps)": int(row["Throughput (flows/s)"]),
        })

    # Map model rows from Attack 34
    for _, row in df_att.iterrows():
        records.append({
            "Phase": "Phase 3: Fine-Grained",
            "Classes": 34,
            "Features": 39,
            "Model": row["Model"],
            "Accuracy (%)": round(row["Accuracy"] * 100, 2),
            "Macro F1": round(row["Macro F1"], 4),
            "Weighted F1": round(row["Weighted F1"], 4),
            "Latency (us/flow)": round(row["Latency (us/flow)"], 2),
            "Throughput (fps)": int(row["Throughput (flows/s)"]),
        })

    # Map reduced features rows from Step 5
    red_20_rows = df_red[df_red["Feature Set"].str.contains("Reduced", case=False)]
    for _, row in red_20_rows.iterrows():
        records.append({
            "Phase": "Step 5: Optimized Edge",
            "Classes": 34,
            "Features": 20,
            "Model": row["Model"],
            "Accuracy (%)": row["Accuracy (%)"],
            "Macro F1": row["Macro F1"],
            "Weighted F1": row["Weighted F1"],
            "Latency (us/flow)": row["Latency (us/flow)"],
            "Throughput (fps)": int(row["Throughput (fps)"]),
        })

    master_df = pd.DataFrame(records)

    # 2. Executive Synthesis by Top Model (XGBoost)
    xgb_subset = master_df[master_df["Model"] == "xgboost"].copy()
    exec_df = xgb_subset[["Phase", "Classes", "Features", "Accuracy (%)", "Macro F1", "Latency (us/flow)", "Throughput (fps)"]]

    # Save to tables
    master_csv = TABLES_DIR / "capstone_master_comparison.csv"
    master_json = TABLES_DIR / "capstone_master_comparison.json"
    exec_csv = TABLES_DIR / "capstone_executive_trajectory.csv"

    master_df.to_csv(master_csv, index=False)
    with open(master_json, "w") as f:
        json.dump(records, f, indent=2)
    exec_df.to_csv(exec_csv, index=False)

    print("=================================================================", flush=True)
    print(" CAPSTONE MASTER SYNTHESIS COMPARISON", flush=True)
    print("=================================================================", flush=True)
    print(tabulate(master_df, headers="keys", tablefmt="pipe", showindex=False), flush=True)

    print("\n=================================================================", flush=True)
    print(" EXECUTIVE ARCHITECTURAL TRAJECTORY (Top Model: XGBoost)", flush=True)
    print("=================================================================", flush=True)
    print(tabulate(exec_df, headers="keys", tablefmt="pipe", showindex=False), flush=True)

    # Generate Publication 4-Panel Visualization
    plot_master_synthesis_figure(master_df)

    return master_df, exec_df


def plot_master_synthesis_figure(master_df: pd.DataFrame) -> None:
    """
    Renders 4-panel publication-ready synthesis figure covering accuracy progression,
    Macro F1 resilience, Pareto efficiency frontier, and IoT edge line-rate throughput.
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    phases_order = [
        "Phase 1: Binary",
        "Phase 2: Category",
        "Phase 3: Fine-Grained",
        "Step 5: Optimized Edge",
    ]

    # Panel 1: Accuracy Progression Across Granularities
    sns.lineplot(
        data=master_df,
        x="Phase",
        y="Accuracy (%)",
        hue="Model",
        style="Model",
        markers=True,
        dashes=False,
        markersize=9,
        lw=2.5,
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("A. Classification Accuracy Across Granularities", fontsize=13, fontweight="bold", pad=10)
    axes[0, 0].set_ylabel("Accuracy (%)", fontsize=11)
    axes[0, 0].set_xlabel("")
    axes[0, 0].set_ylim([60, 100])
    axes[0, 0].tick_params(axis="x", rotation=15)
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)
    axes[0, 0].legend(title="Model", loc="lower left")

    # Panel 2: Macro F1 Progression
    sns.lineplot(
        data=master_df,
        x="Phase",
        y="Macro F1",
        hue="Model",
        style="Model",
        markers=True,
        dashes=False,
        markersize=9,
        lw=2.5,
        ax=axes[0, 1],
    )
    axes[0, 1].set_title("B. Macro F1-Score Progression (Class Imbalance Resilience)", fontsize=13, fontweight="bold", pad=10)
    axes[0, 1].set_ylabel("Macro F1-Score", fontsize=11)
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_ylim([0.45, 0.90])
    axes[0, 1].tick_params(axis="x", rotation=15)
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)
    axes[0, 1].legend(title="Model", loc="lower left")

    # Panel 3: Latency vs. Macro F1 Pareto Frontier
    palette = sns.color_palette("tab10", n_colors=len(master_df["Model"].unique()))
    sns.scatterplot(
        data=master_df,
        x="Latency (us/flow)",
        y="Macro F1",
        hue="Model",
        style="Phase",
        s=180,
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("C. Latency vs. Macro F1 Pareto Frontier", fontsize=13, fontweight="bold", pad=10)
    axes[1, 0].set_xlabel("Per-Flow Inference Latency (us/flow) [Lower is Better]", fontsize=11)
    axes[1, 0].set_ylabel("Macro F1-Score [Higher is Better]", fontsize=11)
    axes[1, 0].set_xscale("log")
    axes[1, 0].grid(True, linestyle=":", alpha=0.6, which="both")

    # Annotate key frontier points
    for _, r in master_df.iterrows():
        if r["Model"] in ["xgboost", "logistic_regression"] or r["Phase"] == "Step 5: Optimized Edge":
            axes[1, 0].annotate(
                f"{r['Model'][:3]}-{r['Phase'].split(':')[0]}",
                xy=(r["Latency (us/flow)"], r["Macro F1"]),
                xytext=(5, 3),
                textcoords="offset points",
                fontsize=7.5,
            )

    # Panel 4: Network Throughput vs Granularity (with IoT Line-Rate thresholds)
    sns.barplot(
        data=master_df,
        x="Phase",
        y="Throughput (fps)",
        hue="Model",
        ax=axes[1, 1],
    )
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("D. Model Throughput & IoT Line-Rate Viability", fontsize=13, fontweight="bold", pad=10)
    axes[1, 1].set_ylabel("Throughput (Flows per Second - Log Scale)", fontsize=11)
    axes[1, 1].set_xlabel("")
    axes[1, 1].tick_params(axis="x", rotation=15)
    axes[1, 1].grid(True, linestyle=":", alpha=0.6, which="both")

    # Line-rate threshold annotations
    axes[1, 1].axhline(y=100_000, color="#e74c3c", linestyle="--", lw=1.5, label="1 Gbps Line-Rate (~100k fps)")
    axes[1, 1].axhline(y=1_000_000, color="#27ae60", linestyle=":", lw=1.5, label="10 Gbps Line-Rate (~1M fps)")
    axes[1, 1].legend(loc="upper right", fontsize=8.5)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "capstone_master_synthesis.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved master synthesis figure to: {fig_path}", flush=True)


if __name__ == "__main__":
    generate_master_synthesis()
