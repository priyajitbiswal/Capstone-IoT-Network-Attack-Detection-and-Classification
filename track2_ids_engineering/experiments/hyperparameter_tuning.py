"""
Research Improvement 2: Hyperparameter Capacity & Regularization Optimization.
Investigates the architectural parameters identified in the 'Optimized Gradient Boosting Framework' paper:
1. Baseline: num_leaves=31 (depth ~5), n_estimators=80, colsample=1.0, subsample=1.0
2. Configuration A (Medium Depth): num_leaves=63 (depth ~7), n_estimators=120, colsample=0.85, subsample=0.85
3. Configuration B (Paper Optimal): num_leaves=127 (depth ~9), n_estimators=150, colsample=0.8, subsample=0.8
4. Configuration C (XGBoost Hist Paper Optimal): max_depth=8, n_estimators=150, colsample=0.8, subsample=0.8

Runs fast (<35 seconds total) on the 100k stratified subset with sqrt_balanced weighting.
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

from src.config import FEATURE_COLUMNS, TARGET_COLUMN, THIRTY_FOUR_CLASSES, THIRTY_FOUR_TO_IDX

SAMPLE_PATH = BASE_DIR.parent / "data" / "processed" / "sample_stratified.csv"
RESULTS_PATH = BASE_DIR / "results" / "tables" / "exp2_hyperparameter_tuning.csv"


def get_sqrt_balanced_weights(y_train):
    classes = np.unique(y_train)
    n_samples = len(y_train)
    n_classes = len(classes)
    class_counts = pd.Series(y_train).value_counts().to_dict()
    raw_weights = {c: np.sqrt(n_samples / (n_classes * class_counts[c])) for c in classes}
    mean_w = np.mean(list(raw_weights.values()))
    norm_weights = {c: w / mean_w for c, w in raw_weights.items()}
    return np.array([norm_weights[cls] for cls in y_train])


def run_experiment_2(sample_size=100_000, random_state=42):
    print("=" * 80)
    print(" EXPERIMENT 2: HYPERPARAMETER TUNING & TREE CAPACITY EXPANSION")
    print("=" * 80)

    # 1. Load Data
    print(f"[*] Loading data sample ({sample_size:,} stratified subset)...")
    df = pd.read_csv(SAMPLE_PATH)

    if len(df) > sample_size:
        df, _ = train_test_split(
            df, train_size=sample_size, stratify=df[TARGET_COLUMN], random_state=random_state
        )
        df = df.reset_index(drop=True)

    # Clean numeric features
    for col in FEATURE_COLUMNS:
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

    X = df[FEATURE_COLUMNS].values
    y = df["label_idx"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    sample_weight = get_sqrt_balanced_weights(y_train)

    configs = [
        {
            "name": "Baseline (LGBM leaves=31, n=80)",
            "type": "lgbm",
            "params": {"num_leaves": 31, "n_estimators": 80, "learning_rate": 0.08, "colsample_bytree": 1.0, "subsample": 1.0},
        },
        {
            "name": "LGBM Config A (leaves=63, n=120, colsample=0.85)",
            "type": "lgbm",
            "params": {"num_leaves": 63, "n_estimators": 120, "learning_rate": 0.08, "colsample_bytree": 0.85, "subsample": 0.85, "subsample_freq": 1},
        },
        {
            "name": "LGBM Config B (leaves=127, n=150, colsample=0.80)",
            "type": "lgbm",
            "params": {"num_leaves": 127, "n_estimators": 150, "learning_rate": 0.08, "colsample_bytree": 0.80, "subsample": 0.80, "subsample_freq": 1},
        },
        {
            "name": "XGBoost Hist Optimal (depth=8, n=120, colsample=0.80)",
            "type": "xgb",
            "params": {"max_depth": 8, "n_estimators": 120, "learning_rate": 0.08, "colsample_bytree": 0.80, "subsample": 0.80, "tree_method": "hist"},
        },
    ]

    results = []

    for cfg in configs:
        print(f"\n--> Training [{cfg['name']}]...", flush=True)
        t0 = time.time()

        if cfg["type"] == "lgbm":
            model = lgb.LGBMClassifier(
                **cfg["params"],
                random_state=random_state,
                n_jobs=-1,
                verbose=-1,
            )
            model.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            model = xgb.XGBClassifier(
                **cfg["params"],
                random_state=random_state,
                n_jobs=-1,
                eval_metric="mlogloss",
                objective="multi:softprob",
                num_class=34,
            )
            model.fit(X_train, y_train, sample_weight=sample_weight)

        train_time = time.time() - t0

        t0_inf = time.time()
        y_pred = model.predict(X_test)
        inf_time_us = ((time.time() - t0_inf) / len(X_test)) * 1_000_000

        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)

        results.append({
            "Configuration": cfg["name"],
            "Accuracy": acc,
            "Macro F1": f1_macro,
            "Weighted F1": f1_weighted,
            "Macro Precision": prec_macro,
            "Macro Recall": rec_macro,
            "Train Time (s)": round(train_time, 2),
            "Latency (us/flow)": round(inf_time_us, 2),
        })

    df_res = pd.DataFrame(results)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(RESULTS_PATH, index=False)

    print("\n" + "=" * 80)
    print(" EXPERIMENT 2 RESULTS SUMMARY:")
    print("=" * 80)
    headers = ["Configuration", "Accuracy", "Macro F1", "Weighted F1", "Macro Prec", "Macro Rec", "Train(s)", "Lat(us)"]
    table_data = [
        [
            r["Configuration"],
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
    run_experiment_2()
