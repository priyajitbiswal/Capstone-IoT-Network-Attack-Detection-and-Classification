"""
Feature Decay Sweep:
Evaluates classification accuracy and Macro F1 as the feature space is systematically
reduced from 39 features down to 20 (and below), pruning 3-4 features at each step
based on the consensus importance ranking.

Optimized for fast, representative execution (<30 seconds total runtime).
"""

import sys
from pathlib import Path
import time

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import TARGET_COLUMN

# Paths
SAMPLE_PATH = BASE_DIR / "data" / "sample_stratified.csv"
RANKINGS_PATH = BASE_DIR / "results" / "tables" / "feature_importance_rankings.csv"
OUTPUT_TABLE = BASE_DIR / "results" / "tables" / "feature_decay_sweep.csv"
OUTPUT_FIG = BASE_DIR / "results" / "figures" / "feature_decay_curve.png"


def run_feature_decay_sweep(
    sample_size: int = 100_000,
    steps: list = [39, 36, 32, 28, 24, 20, 16, 12, 8, 4, 2],
    random_state: int = 42,
):
    print("=" * 80)
    print(" FEATURE DECAY SWEEP: ACCURACY VS FEATURE COUNT (39 down to 2)")
    print("=" * 80)

    # 1. Load Feature Rankings
    if not RANKINGS_PATH.exists():
        raise FileNotFoundError(f"Rankings file not found: {RANKINGS_PATH}")
    
    df_rankings = pd.read_csv(RANKINGS_PATH)
    ranked_features = df_rankings["Feature"].tolist()
    cum_scores = df_rankings["Cumulative_Score"].tolist()
    print(f"[*] Loaded consensus rankings for {len(ranked_features)} features.")

    # 2. Load and Prepare Sample Data
    print(f"[*] Loading data sample ({sample_size:,} stratified subset for fast evaluation)...")
    t0_data = time.time()
    df = pd.read_csv(SAMPLE_PATH)

    # Clean numeric features
    for col in ranked_features:
        inf_mask = np.isinf(df[col])
        if inf_mask.any():
            finite_max = df.loc[~inf_mask, col].max()
            df.loc[inf_mask, col] = finite_max
        if df[col].isna().any():
            df[col] = df[col].fillna(0.0)

    # Integer encode 34-class labels
    y_raw = df[TARGET_COLUMN].astype(str).str.strip().str.upper()
    classes = sorted(y_raw.unique())
    label_to_idx = {c: i for i, c in enumerate(classes)}
    y = y_raw.map(label_to_idx).values.astype(np.int64)

    # Subsample stratified for speed
    if len(df) > sample_size:
        _, df_sub, _, y_sub = train_test_split(
            df, y, test_size=sample_size, random_state=random_state, stratify=y
        )
    else:
        df_sub, y_sub = df, y

    # Fixed 80/20 train/test split
    idx_train, idx_test = train_test_split(
        np.arange(len(df_sub)),
        test_size=0.2,
        random_state=random_state,
        stratify=y_sub,
    )
    y_train = y_sub[idx_train]
    y_test = y_sub[idx_test]
    print(f"[*] Data ready in {time.time() - t0_data:.2f}s (Train: {len(y_train):,}, Test: {len(y_test):,}).\n")

    results = []
    baseline_acc = None
    baseline_f1 = None

    print("-" * 88)
    print(f"{'Step':<5} | {'Feats':<5} | {'Cum Info':<9} | {'Accuracy':<10} | {'Macro F1':<9} | {'Acc Delta':<10} | {'Pruned Features in Step'}")
    print("-" * 88)

    for i, k in enumerate(steps, 1):
        t_step = time.time()
        # Top-k features according to ranking
        selected_cols = ranked_features[:k]
        cum_info = cum_scores[k - 1] * 100

        # Features pruned since previous step
        if i == 1:
            pruned_desc = "None (Full Baseline Set)"
        else:
            prev_k = steps[i - 2]
            pruned_cols = ranked_features[k:prev_k]
            pruned_desc = f"-{len(pruned_cols)}: " + ", ".join(pruned_cols[:3])
            if len(pruned_cols) > 3:
                pruned_desc += f", +{len(pruned_cols)-3} more"

        X_mat = df_sub[selected_cols].values.astype(np.float32)
        X_tr = X_mat[idx_train]
        X_te = X_mat[idx_test]

        # Standard scale
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        # Fast Random Forest (40 estimators, max_depth 14)
        clf = RandomForestClassifier(
            n_estimators=40,
            max_depth=14,
            n_jobs=-1,
            random_state=random_state,
        )
        clf.fit(X_tr, y_train)
        preds = clf.predict(X_te)

        acc = accuracy_score(y_test, preds) * 100
        macro_f1 = f1_score(y_test, preds, average="macro", zero_division=0)

        if baseline_acc is None:
            baseline_acc = acc
            baseline_f1 = macro_f1
            acc_delta_str = "Baseline"
            acc_delta = 0.0
        else:
            acc_delta = acc - baseline_acc
            acc_delta_str = f"{acc_delta:+.2f}%"

        duration = time.time() - t_step

        print(
            f"{i:<5} | {k:<5} | {cum_info:>6.1f}%   | {acc:>6.2f}%    | {macro_f1:>8.4f}  | {acc_delta_str:>10} | {pruned_desc} ({duration:.1f}s)"
        )

        results.append({
            "Step": i,
            "Feature_Count": k,
            "Cumulative_Info_Pct": round(cum_info, 2),
            "Accuracy_Pct": round(acc, 2),
            "Macro_F1": round(macro_f1, 4),
            "Acc_Delta_Pct": round(acc_delta, 2),
            "Features_List": "; ".join(selected_cols),
        })

    print("-" * 88)

    # 3. Save Summary Table
    df_results = pd.DataFrame(results)
    OUTPUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(OUTPUT_TABLE, index=False)
    print(f"\n[+] Saved complete decay sweep table to: {OUTPUT_TABLE}")

    # 4. Generate & Save Publication Plot
    plt.figure(figsize=(10, 6))
    plt.plot(
        df_results["Feature_Count"],
        df_results["Accuracy_Pct"],
        marker="o",
        color="#2980b9",
        linewidth=2.5,
        label="34-Class Accuracy (%)",
    )
    plt.axvline(
        x=20,
        color="#e74c3c",
        linestyle="--",
        linewidth=2,
        label="Optimal Edge Cutoff (Top 20 Features)",
    )
    plt.annotate(
        f"20 Feats: {df_results[df_results['Feature_Count']==20]['Accuracy_Pct'].values[0]:.2f}%\n(Safe Plateau)",
        xy=(20, df_results[df_results["Feature_Count"] == 20]["Accuracy_Pct"].values[0]),
        xytext=(22, df_results[df_results["Feature_Count"] == 20]["Accuracy_Pct"].values[0] - 4),
        arrowprops=dict(facecolor="#e74c3c", shrink=0.08, width=1.5, headwidth=6),
        fontweight="bold",
        color="#c0392b",
    )

    plt.title("Feature Decay Curve: Accuracy vs Feature Count", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Number of Selected Features (Ranked by Importance)", fontsize=11)
    plt.ylabel("Classification Accuracy (%)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.xlim(0, 41)
    plt.ylim(
        min(df_results["Accuracy_Pct"]) - 5,
        max(df_results["Accuracy_Pct"]) + 3,
    )
    plt.legend(loc="lower right")
    plt.tight_layout()

    OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_FIG, dpi=300)
    plt.close()
    print(f"[+] Saved decay curve visualization to: {OUTPUT_FIG}\n")

    return df_results


if __name__ == "__main__":
    run_feature_decay_sweep()
