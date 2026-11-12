"""
Generates publication-ready matplotlib figures for Track 2 Minority-Class Optimization
Experiments 1-3 across ORIG34 and OP29 taxonomies.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from track1_dataset_evaluation.src.config import THIRTY_FOUR_CLASSES
from track2_ids_engineering.hierarchical_ids.src.data_utils import FLOOD_CLASSES, MINORITY_ATTACK_CLASSES
from track2_ids_engineering.minority_optimization.src.common import (
    CM_DIR,
    CONFIGS_DIR,
    FIGURES_DIR,
    OP29_CLASSES,
    OP29_FLOOD_CLASSES,
    RESULTS_DIR,
)


plt.rcParams.update(
    {
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.0,
        "figure.dpi": 200,
    }
)


def plot_smote_realism_audit() -> None:
    df = pd.read_csv(RESULTS_DIR / "exp1_smote_realism_audit.csv")
    df = df.sort_values("train_count", ascending=True).reset_index(drop=True)
    y = np.arange(len(df))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)

    h = 0.36
    ax1.barh(y - h / 2, df["real_pct_seeds_ge1_same_5nn"], height=h, color="#1f77b4", label=">=1 Same-Class 5-NN (%)")
    ax1.barh(y + h / 2, df["real_pct_seeds_majority_benign_5nn"], height=h, color="#d62728", alpha=0.85, label="Majority BENIGN 5-NN (%)")
    ax1.axvline(15.0, color="#333333", linestyle="--", linewidth=1.0, label="Selective 15% Gate")
    ax1.set_yticks(y)
    ax1.set_yticklabels([f"{c} (n={n})" for c, n in zip(df["class_34"], df["train_count"])])
    ax1.set_xlabel("Percentage of Training Seeds (%)")
    ax1.set_title("Real Train 5-NN Neighborhood Purity vs BENIGN Overlap")
    ax1.set_xlim(0, 100)
    ax1.grid(axis="x", alpha=0.3)
    ax1.legend(loc="lower right")

    h2 = 0.25
    ax2.barh(y - h2, df["std_proto_pair_mismatch_pct"], height=h2, color="#ff7f0e", label="Std SMOTE Mixed Protocol Pairs (%)")
    ax2.barh(y, df["std_variance_identity_err_pct"], height=h2, color="#9467bd", label="Std SMOTE Broken Variance=Std^2 (%)")
    ax2.barh(y + h2, df["std_syn_1nn_wrong_class_pct"], height=h2, color="#2ca02c", alpha=0.85, label="Std SMOTE Syn 1-NN != Own Class (%)")
    ax2.set_xlabel("Violation / Collision Rate on Synthetic Samples (%)")
    ax2.set_title("Standard Chawla SMOTE Feature & Manifold Violations")
    ax2.set_xlim(0, 105)
    ax2.grid(axis="x", alpha=0.3)
    ax2.legend(loc="lower right")

    fig.suptitle("Experiment 1 Pre-Training Audit: Minority Neighborhood Purity & SMOTE Realism (Train n=350,821)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_smote_realism_audit.png", dpi=200)
    plt.close(fig)


def plot_val_sweeps() -> None:
    df_base = pd.read_csv(RESULTS_DIR / "phase1_baseline_e4.csv")
    df_s34 = pd.read_csv(RESULTS_DIR / "exp1_smote_val_sweep_34.csv")
    df_s29 = pd.read_csv(RESULTS_DIR / "exp1_smote_val_sweep_op29.csv")
    df_w34 = pd.read_csv(RESULTS_DIR / "exp2_weights_val_sweep_34.csv")
    df_w29 = pd.read_csv(RESULTS_DIR / "exp2_weights_val_sweep_op29.csv")

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.2))

    # Top-left: Exp1 SMOTE ORIG34 & OP29
    for ax, df_s, base_m, title in [
        (axes[0, 0], df_s34, "E4_Baseline_34", "Exp 1 (SMOTE): Validation Sweep — ORIG34"),
        (axes[0, 1], df_s29, "E4_Baseline_OP29_Remap", "Exp 1 (SMOTE): Validation Sweep — OP29"),
    ]:
        b = df_base[(df_base["model"] == base_m) & (df_base["split"] == "val")].iloc[0]
        labels = [m.replace("SMOTE_", "") for m in df_s["smote_mode"]]
        x = np.arange(len(labels))
        w = 0.25
        ax.bar(x - w, df_s["macro_f1"], width=w, color="#1f77b4", label="Macro-F1")
        ax.bar(x, df_s["minority_f1"], width=w, color="#2ca02c", label="Minority F1")
        ax.bar(x + w, df_s["benign_fpr"], width=w, color="#d62728", alpha=0.8, label="Benign FPR")
        ax.axhline(b["macro_f1"], color="#1f77b4", linestyle="--", linewidth=1.0, label="E4 Base Macro-F1")
        ax.axhline(b["minority_f1"], color="#2ca02c", linestyle=":", linewidth=1.3, label="E4 Base Min-F1")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15)
        ax.set_ylim(0, 0.85)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(ncol=2, fontsize=7.5)

    # Bottom: Exp2 Weighting Alpha curves ORIG34 & OP29
    for ax, df_w, base_m, title in [
        (axes[1, 0], df_w34, "E4_Baseline_34", "Exp 2 (Class Weights): Validation Alpha Curve — ORIG34"),
        (axes[1, 1], df_w29, "E4_Baseline_OP29_Remap", "Exp 2 (Class Weights): Validation Alpha Curve — OP29"),
    ]:
        df_w = df_w.sort_values("weight_alpha")
        b = df_base[(df_base["model"] == base_m) & (df_base["split"] == "val")].iloc[0]
        alphas = df_w["weight_alpha"]
        ax.plot(alphas, df_w["accuracy"], "o-", color="#444444", label="Accuracy")
        ax.plot(alphas, df_w["macro_f1"], "s-", color="#1f77b4", label="Macro-F1")
        ax.plot(alphas, df_w["minority_f1"], "^-", color="#2ca02c", label="Minority F1")
        ax.plot(alphas, df_w["minority_recall"], "v--", color="#9467bd", label="Minority Recall")
        ax.plot(alphas, df_w["benign_fpr"], "d:", color="#d62728", label="Benign FPR")
        ax.axhline(b["macro_f1"], color="#1f77b4", linestyle=":", alpha=0.5)
        ax.axhline(b["minority_f1"], color="#2ca02c", linestyle=":", alpha=0.5)
        ax.set_xticks([0.25, 0.50, 0.75, 1.00])
        ax.set_xlabel("Power Weight Exponent (alpha)")
        ax.set_ylim(0.10, 0.92)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(ncol=2, fontsize=7.5)

    fig.suptitle("Validation-Only Parameter & Weight Sweeps (n=75,172; Zero Test/Cross Leakage)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_val_tuning_curves.png", dpi=200)
    plt.close(fig)


def plot_final_comparisons() -> None:
    df34 = pd.read_csv(RESULTS_DIR / "phase4_summary_34.csv")
    df29 = pd.read_csv(RESULTS_DIR / "phase4_summary_op29.csv")

    with open(CONFIGS_DIR / "exp1_selected_smote.json", "r") as f:
        e1 = json.load(f)["selections"]
    with open(CONFIGS_DIR / "exp2_selected_weights.json", "r") as f:
        e2 = json.load(f)["selections"]

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 8.6))
    metrics = ["accuracy", "macro_f1", "minority_f1", "minority_recall", "benign_fpr"]
    m_labels = ["Accuracy", "Macro-F1", "Minority F1", "Minority Rec", "Benign FPR"]

    panels = [
        (axes[0, 0], df34, "ORIG34", "34", "test", "E4_Baseline_34", "ORIG34 — Internal Test (n=75,105)"),
        (axes[0, 1], df34, "ORIG34", "34", "cross", "E4_Baseline_34", "ORIG34 — Cross-Capture (n=57,280)"),
        (axes[1, 0], df29, "OP29", "op29", "test", "E4_Baseline_OP29_Remap", "OP29 — Internal Test (n=75,105)"),
        (axes[1, 1], df29, "OP29", "op29", "cross", "E4_Baseline_OP29_Remap", "OP29 — Cross-Capture (n=57,280)"),
    ]

    for ax, df, tax, short, sp, base_name, title in panels:
        sub_names = [
            (base_name, "E4 Base"),
            (f"Exp1_SMOTE_Std_Cap4000_{short}", "Exp1 Std"),
            (e1[tax]["selected_model"], "Exp1 Best"),
            (e2[tax]["selected_model"], "Exp2 Best"),
            (f"Exp3_Hybrid_Naive_{short}", "Exp3 Naive"),
            (f"Exp3_Hybrid_Gated_{short}", "Exp3 Gated"),
        ]
        # Deduplicate while preserving order if Exp1 Best == Exp1 Std
        seen = set()
        models_ordered = []
        for mn, lbl in sub_names:
            if mn not in seen:
                seen.add(mn)
                models_ordered.append((mn, lbl))

        x = np.arange(len(metrics))
        w = 0.82 / len(models_ordered)
        colors = ["#333333", "#ff7f0e", "#2ca02c", "#1f77b4", "#d62728", "#9467bd"]
        for i, (mn, lbl) in enumerate(models_ordered):
            row = df[(df["model"] == mn) & (df["split"] == sp)].iloc[0]
            vals = [row[m] for m in metrics]
            ax.bar(x + (i - (len(models_ordered) - 1) / 2.0) * w, vals, width=w, color=colors[i % len(colors)], label=lbl)

        ax.set_xticks(x)
        ax.set_xticklabels(m_labels)
        ax.set_ylim(0, 0.95)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(ncol=3, fontsize=7.5)

    fig.suptitle("Final Method Comparison Across Internal Test & Cross-Capture (ORIG34 vs OP29)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_taxonomy_comparison_34_vs_op29.png", dpi=200)
    plt.close(fig)


def plot_minority_heatmap() -> None:
    pc34 = pd.read_csv(RESULTS_DIR / "phase4_per_class_test_34.csv")
    pc29 = pd.read_csv(RESULTS_DIR / "phase4_per_class_test_op29.csv")

    with open(CONFIGS_DIR / "exp1_selected_smote.json", "r") as f:
        e1 = json.load(f)["selections"]
    with open(CONFIGS_DIR / "exp2_selected_weights.json", "r") as f:
        e2 = json.load(f)["selections"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 5.5))

    for ax, pc, tax, short, base_m, title in [
        (ax1, pc34, "ORIG34", "34", "E4_Baseline_34", "34-Class Test F1 Across 8 Minority Classes"),
        (ax2, pc29, "OP29", "op29", "E4_Baseline_OP29_Remap", "OP29 Test F1 Across 8 Minority Classes"),
    ]:
        model_list = [
            (base_m, "E4 Base"),
            (f"Exp1_SMOTE_Std_Cap4000_{short}", "Exp1 StdSMOTE"),
            (e1[tax]["selected_model"], "Exp1 BestSMOTE"),
            (f"Exp2_Power_a025_{short}", "Exp2 a=0.25"),
            (f"Exp2_SqrtBalanced_a050_{short}", "Exp2 a=0.50"),
            (f"Exp2_Power_a075_{short}", "Exp2 a=0.75"),
            (f"Exp2_Balanced_a100_{short}", "Exp2 a=1.00"),
            (f"Exp3_Hybrid_Naive_{short}", "Exp3 Naive"),
            (f"Exp3_Hybrid_Gated_{short}", "Exp3 Gated"),
        ]
        # Unique ordered
        seen = set()
        m_filt = []
        for mn, lbl in model_list:
            if mn not in seen:
                seen.add(mn)
                m_filt.append((mn, lbl))

        mat = np.zeros((len(MINORITY_ATTACK_CLASSES), len(m_filt)), dtype=float)
        for j, (mn, _) in enumerate(m_filt):
            sub = pc[pc["model"] == mn].set_index("class")
            for i, c in enumerate(MINORITY_ATTACK_CLASSES):
                mat[i, j] = float(sub.loc[c, "f1"])

        im = ax.imshow(mat, cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=0.60)
        ax.set_yticks(np.arange(len(MINORITY_ATTACK_CLASSES)))
        ax.set_yticklabels(MINORITY_ATTACK_CLASSES)
        ax.set_xticks(np.arange(len(m_filt)))
        ax.set_xticklabels([lbl for _, lbl in m_filt], rotation=28, ha="right")
        ax.set_title(title)

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", color="white" if val > 0.34 else "black", fontsize=7.5)
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)

    fig.suptitle("Per-Class F1 Breakdown on the 8 Rare Non-Flood Minority Classes (Internal Test n=75,105)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_minority_per_class_f1_heatmap.png", dpi=200)
    plt.close(fig)


def plot_exp3_policy_frontier() -> None:
    df34 = pd.read_csv(RESULTS_DIR / "exp3_policy_val_sweep_34.csv")
    df29 = pd.read_csv(RESULTS_DIR / "exp3_policy_val_sweep_op29.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.2))
    for ax, df, title in [
        (ax1, df34, "ORIG34: Specialist Validation Policy Frontier"),
        (ax2, df29, "OP29: Specialist Validation Policy Frontier"),
    ]:
        # Slice tau_cls=0.0, tau_e4_keep=1.01 across tau_rare to show pure tau_rare sensitivity
        curve = df[(df["policy_kind"] == "GridSearch") & (df["tau_cls"] == 0.0) & (df["tau_e4_keep"] > 1.0)].sort_values("tau_rare")
        grid = df[df["policy_kind"] == "GridSearch"]
        naive = df[df["policy_kind"] == "Naive_tau050"].iloc[0]

        ax.scatter(grid["benign_fpr"], grid["minority_f1"], c=grid["macro_f1"], cmap="viridis", alpha=0.65, s=28, label="Policy Grid")
        ax.plot(curve["benign_fpr"], curve["minority_f1"], "k--", linewidth=1.2, alpha=0.8, label="tau_rare sweep (unconditional E4 override)")
        ax.scatter([naive["benign_fpr"]], [naive["minority_f1"]], color="red", marker="X", s=110, zorder=5, label="Naive tau=0.50")

        ax.set_xlabel("Validation Benign False Positive Rate (Benign FPR)")
        ax.set_ylabel("Validation Minority F1")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8.0)

    fig.suptitle("Experiment 3: Why Naive tau=0.50 Fails (Benign FPR Explosion) vs Gated Specialist Thresholds", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_exp3_specialist_tradeoff.png", dpi=200)
    plt.close(fig)


def plot_minority_confusion_blocks() -> None:
    with open(CONFIGS_DIR / "exp1_selected_smote.json", "r") as f:
        e1 = json.load(f)["selections"]
    with open(CONFIGS_DIR / "exp2_selected_weights.json", "r") as f:
        e2 = json.load(f)["selections"]

    for tax, short, cls_names, flood_list, base_m in [
        ("ORIG34", "34", THIRTY_FOUR_CLASSES, FLOOD_CLASSES, "E4_Baseline_34"),
        ("OP29", "op29", OP29_CLASSES, OP29_FLOOD_CLASSES, "E4_Baseline_OP29_Remap"),
    ]:
        rows_cls = ["BENIGN"] + list(MINORITY_ATTACK_CLASSES)
        row_idx = [cls_names.index(c) for c in rows_cls]
        flood_idx = np.array([cls_names.index(c) for c in flood_list], dtype=int)
        other_nf_idx = np.array([i for i, c in enumerate(cls_names) if c not in rows_cls and c not in flood_list], dtype=int)

        col_labels = ["BENIGN"] + [c[:10] for c in MINORITY_ATTACK_CLASSES] + ["OtherNF", "Flood"]

        models = [
            (base_m, "E4 Baseline"),
            (e1[tax]["selected_model"], "Exp1 Best SMOTE"),
            (e2[tax]["selected_model"], "Exp2 Best Weight"),
            (f"Exp3_Hybrid_Gated_{short}", "Exp3 Hybrid Gated"),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(15.5, 11.2))
        for ax, (mn, lbl) in zip(axes.flat, models):
            cm_path = CM_DIR / f"{mn}_test_{short}.npy"
            if not cm_path.exists():
                continue
            cm = np.load(cm_path)
            sub = np.zeros((len(row_idx), len(col_labels)), dtype=float)
            for ri, r_global in enumerate(row_idx):
                row_counts = cm[r_global]
                tot = max(1, int(np.sum(row_counts)))
                for ci, c_global in enumerate(row_idx):
                    sub[ri, ci] = row_counts[c_global] / tot * 100.0
                sub[ri, len(row_idx)] = np.sum(row_counts[other_nf_idx]) / tot * 100.0
                sub[ri, len(row_idx) + 1] = np.sum(row_counts[flood_idx]) / tot * 100.0

            ax.imshow(sub, cmap="Blues", aspect="auto", vmin=0.0, vmax=80.0)
            ax.set_yticks(np.arange(len(rows_cls)))
            ax.set_yticklabels(rows_cls)
            ax.set_xticks(np.arange(len(col_labels)))
            ax.set_xticklabels(col_labels, rotation=32, ha="right")
            ax.set_title(f"{lbl} ({mn})")

            for i in range(sub.shape[0]):
                for j in range(sub.shape[1]):
                    v = sub[i, j]
                    if v >= 0.5:
                        ax.text(j, i, f"{v:.1f}", ha="center", va="center", color="white" if v > 42 else "black", fontsize=7.0)

        fig.suptitle(f"{tax} Internal Test Row-Normalized Routing (%): BENIGN + 8 Minority Classes", fontweight="bold")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"fig_{short}_minority_confusion_blocks.png", dpi=200)
        plt.close(fig)


def main() -> None:
    print("Generating Track 2 minority optimization figures...")
    plot_smote_realism_audit()
    plot_val_sweeps()
    plot_final_comparisons()
    plot_minority_heatmap()
    plot_exp3_policy_frontier()
    plot_minority_confusion_blocks()
    print(f"Saved all figures under {FIGURES_DIR}")


if __name__ == "__main__":
    main()
