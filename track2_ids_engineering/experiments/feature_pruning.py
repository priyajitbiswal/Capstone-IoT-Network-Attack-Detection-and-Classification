"""
Research Improvement 3: Feature Space Pruning (Top 23 Features vs 39 Features).
Evaluates whether removing 16 low-variance/noisy protocol flags improves generalization,
prevents overfitting, and reduces inference latency, confirming the findings of the
'Optimized Gradient Boosting Framework' paper.

Runs in ~25 seconds on the 100k stratified subset.
"""

import sys
from pathlib import Path
import time

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import lightgbm as lgb
import xgboost as xgb
from tabulate import tabulate

from src.config import TARGET_COLUMN, THIRTY_FOUR_CLASSES, THIRTY_FOUR_TO_IDX

SAMPLE_PATH = BASE_DIR.parent / "data" / "processed" / "sample_stratified.csv"
RANKINGS_PATH = BASE_DIR / "results" / "tables" / "feature_importance_rankings.csv"
RESULTS_PATH = BASE_DIR / "results" / "tables" / "exp3_feature_pruning.csv"


def get_sqrt_balanced_weights(y_train):
    classes = np.unique(y_train)
    n_samples = len(y_train)
    n_classes = len(classes)
    class_counts = pd.Series(y_train).value_counts().to_dict()
    raw_weights = {c: np.sqrt(n_samples / (n_classes * class_counts[c])) for c in classes}
    mean_w = np.mean(list(raw_weights.values()))
    norm_weights = {c: w / mean_w for c, w in raw_weights.items()}
    return np.array([norm_weights[cls] for cls in y_train])


def run_experiment_3(sample_size=100_000, random_state=42):
    print("=" * 80)
    print(" EXPERIMENT 3: FEATURE PRUNING (TOP 23 VS 39 FEATURES)")
    print("=" * 80)

    # 1. Load Rankings
    df_rankings = pd.read_csv(RANKINGS_PATH)
    all_39_features = df_rankings["Feature"].tolist()
    top_23_features = all_39_features[:23]

    print(f"[*] Top 23 Features: {top_23_features}")

    # 2. Load Data
    print(f"[*] Loading data sample ({sample_size:,} stratified subset)...")
    df = pd.read_csv(SAMPLE_PATH)

    if len(df) > sample_size:
        df, _ = train_test_split(
            df, train_size=sample_size, stratify=df[TARGET_COLUMN], random_state=random_state
        )
        df = df.reset_index(drop=True)

    # Clean numeric features
    for col in all_39_features:
        inf_mask = np.isinf(df[col])
        if inf_mask.any():
            median_val = df.loc[~inf_mask, col].median()
            df.loc[inf_mask, col] = median_val
        nan_mask = df[col].isna()
        if nan_mask.any():
            median_val = df.loc[~nan_mask, col].median()
            df.loc[nan_mask, col] = median_val

    df["label_idx"] = df[TARGET_COLUMN].str.upper().map(THIRTY_FOUR_TO_IDX)
    df = df.dropna(subset=["label_idx"])
    df["label_idx"] = df["label_idx"].astype(int)

    feature_sets = [
        ("Full Feature Space (39 features)", all_39_features),
        ("Curated Paper Subset (Top 23 features)", top_23_features),
    ]

    results = []

    for f_name, f_cols in feature_sets:
        print(f"\n--> Evaluating: [{f_name}] ({len(f_cols)} features)...")
        X = df[f_cols].values
        y = df["label_idx"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        sample_weight = get_sqrt_balanced_weights(y_train)

        # 1. Test LightGBM
        t0 = time.time()
        clf_lgb = lgb.LGBMClassifier(
            num_leaves=31,
            n_estimators=100,
            learning_rate=0.08,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
        clf_lgb.fit(X_train, y_train, sample_weight=sample_weight)
        train_time_lgb = time.time() - t0

        t0_inf = time.time()
        y_pred_lgb = clf_lgb.predict(X_test)
        lat_lgb = ((time.time() - t0_inf) / len(X_test)) * 1_000_000

        results.append({
            "Feature Set": f"{f_name} - LightGBM",
            "Features": len(f_cols),
            "Accuracy": accuracy_score(y_test, y_pred_lgb),
            "Macro F1": f1_score(y_test, y_pred_lgb, average="macro", zero_division=0),
            "Weighted F1": f1_score(y_test, y_pred_lgb, average="weighted", zero_division=0),
            "Macro Precision": precision_score(y_test, y_pred_lgb, average="macro", zero_division=0),
            "Macro Recall": recall_score(y_test, y_pred_lgb, average="macro", zero_division=0),
            "Train Time (s)": round(train_time_lgb, 2),
            "Latency (us/flow)": round(lat_lgb, 2),
        })

        # 2. Test XGBoost Hist
        t0_xgb = time.time()
        clf_xgb = xgb.XGBClassifier(
            max_depth=6,
            n_estimators=100,
            learning_rate=0.08,
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
            eval_metric="mlogloss",
            objective="multi:softprob",
            num_class=34,
        )
        clf_xgb.fit(X_train, y_train, sample_weight=sample_weight)
        train_time_xgb = time.time() - t0_xgb

        t0_inf_xgb = time.time()
        y_pred_xgb = clf_xgb.predict(X_test)
        lat_xgb = ((time.time() - t0_inf_xgb) / len(X_test)) * 1_000_000

        results.append({
            "Feature Set": f"{f_name} - XGBoost Hist",
            "Features": len(f_cols),
            "Accuracy": accuracy_score(y_test, y_pred_xgb),
            "Macro F1": f1_score(y_test, y_pred_xgb, average="macro", zero_division=0),
            "Weighted F1": f1_score(y_test, y_pred_xgb, average="weighted", zero_division=0),
            "Macro Precision": precision_score(y_test, y_pred_xgb, average="macro", zero_division=0),
            "Macro Recall": recall_score(y_test, y_pred_xgb, average="macro", zero_division=0),
            "Train Time (s)": round(train_time_xgb, 2),
            "Latency (us/flow)": round(lat_xgb, 2),
        })

    df_res = pd.DataFrame(results)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(RESULTS_PATH, index=False)

    print("\n" + "=" * 80)
    print(" EXPERIMENT 3 RESULTS SUMMARY:")
    print("=" * 80)
    headers = ["Feature Set", "Feat#", "Accuracy", "Macro F1", "Weighted F1", "Macro Prec", "Macro Rec", "Train(s)", "Lat(us)"]
    table_data = [
        [
            r["Feature Set"],
            r["Features"],
            f"{r['Accuracy']*100:.2f}%",
            f"{r['Macro F1']:.4f}",
            f"{r['Weighted F1']:.4f}",
            f"{r['Macro Precision']:.4f}",
            f"{r['Macro Recall']:.4f}",
            f"{r['Train Time (s)']:.1f}",
            f"{r['Latency (us/flow)']:.1f}",
        ]
        for r in results
    ]
    print(tabulate(table_data, headers=headers, tablefmt="github"))
    return df_res


if __name__ == "__main__":
    run_experiment_3()
