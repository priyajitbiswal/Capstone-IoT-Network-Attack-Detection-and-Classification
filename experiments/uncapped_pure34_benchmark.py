"""
Experiment 7: Pure 34-Class Classification with Flood Capping Completely Removed.
Reverts all label merging (keeps all 34 classes strictly distinct and original).
Tests what accuracy our models achieve when trained and tested on the natural,
un-capped traffic stream from MERGED_CSV files (where floods are not capped at 25k).

Runs fast (~30 seconds).
"""

import sys
from pathlib import Path
import time
import glob

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import lightgbm as lgb
import xgboost as xgb
from tabulate import tabulate

from src.config import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    THIRTY_FOUR_CLASSES,
    THIRTY_FOUR_TO_IDX,
    IDX_TO_THIRTY_FOUR,
)

SAMPLE_PATH = BASE_DIR / "data" / "sample_stratified.csv"
RESULTS_PATH = BASE_DIR / "results" / "tables" / "exp7_uncapped_pure34.csv"


def clean_dataframe(df, cols):
    for col in cols:
        inf_mask = np.isinf(df[col])
        if inf_mask.any():
            median_val = df.loc[~inf_mask, col].median()
            df.loc[inf_mask, col] = median_val
        nan_mask = df[col].isna()
        if nan_mask.any():
            median_val = df.loc[~nan_mask, col].median()
            df.loc[nan_mask, col] = median_val
    return df


def load_natural_uncapped_data(n_samples=100_000, max_files=10, random_state=42):
    """
    Loads network flows directly from raw MERGED_CSV files WITHOUT ANY CAPPING on flood classes.
    """
    raw_dir = BASE_DIR / "data" / "raw" if (BASE_DIR / "data" / "raw").exists() else BASE_DIR / "MERGED_CSV"
    csv_files = glob.glob(str(raw_dir / "*.csv"))[:max_files]
    chunks = []
    per_file = n_samples // len(csv_files)
    
    for f in csv_files:
        df_chunk = pd.read_csv(f, nrows=per_file + 500)
        chunks.append(df_chunk)
        
    df_raw = pd.concat(chunks, ignore_index=True)
    if len(df_raw) > n_samples:
        df_raw = df_raw.sample(n=n_samples, random_state=random_state).reset_index(drop=True)
        
    df_raw = clean_dataframe(df_raw, FEATURE_COLUMNS)
    df_raw["label_idx"] = df_raw[TARGET_COLUMN].str.upper().map(THIRTY_FOUR_TO_IDX)
    df_raw = df_raw.dropna(subset=["label_idx"])
    df_raw["label_idx"] = df_raw["label_idx"].astype(int)
    return df_raw


def run_experiment_7(sample_size=100_000, random_state=42):
    print("=" * 80)
    print(" EXPERIMENT 7: PURE 34-CLASS BENCHMARK &mdash; FLOOD CAPPING REMOVED")
    print(" (Zero class merging &mdash; all 34 classes strictly distinct and original)")
    print("=" * 80)

    # 1. Load Capped Stratified Dataset (Our Project Baseline)
    print(f"\n[*] [1/2] Loading Capped Stratified Dataset ({sample_size:,} rows)...")
    df_capped = pd.read_csv(SAMPLE_PATH)
    if len(df_capped) > sample_size:
        df_capped, _ = train_test_split(
            df_capped, train_size=sample_size, stratify=df_capped[TARGET_COLUMN], random_state=random_state
        )
        df_capped = df_capped.reset_index(drop=True)
    df_capped = clean_dataframe(df_capped, FEATURE_COLUMNS)
    df_capped["label_idx"] = df_capped[TARGET_COLUMN].str.upper().map(THIRTY_FOUR_TO_IDX)
    df_capped = df_capped.dropna(subset=["label_idx"])
    df_capped["label_idx"] = df_capped["label_idx"].astype(int)

    # 2. Load Natural Un-Capped Dataset (Raw Stream without any caps)
    print(f"[*] [2/2] Loading Natural Un-Capped Dataset ({sample_size:,} rows from MERGED_CSV)...")
    df_uncapped = load_natural_uncapped_data(n_samples=sample_size, random_state=random_state)
    print(f"    [*] Un-capped total flows: {len(df_uncapped):,}")
    print(f"    [*] Un-capped flood percentage: {(df_uncapped['label_idx'].isin([THIRTY_FOUR_TO_IDX[c] for c in THIRTY_FOUR_CLASSES if 'DDOS' in c or 'DOS' in c])).mean()*100:.1f}%")

    results = []

    # -------------------------------------------------------------
    # Regime A: Capped Dataset (Baseline)
    # -------------------------------------------------------------
    X_cap = df_capped[FEATURE_COLUMNS].values
    y_cap = df_capped["label_idx"].values
    X_cap_tr, X_cap_te, y_cap_tr, y_cap_te = train_test_split(
        X_cap, y_cap, test_size=0.2, random_state=random_state, stratify=y_cap
    )
    scaler_cap = StandardScaler()
    X_cap_tr = scaler_cap.fit_transform(X_cap_tr)
    X_cap_te = scaler_cap.transform(X_cap_te)

    print("\n--> Evaluating Baseline (Capped Stratified Sample, 34 Classes)...", flush=True)
    clf_cap = lgb.LGBMClassifier(num_leaves=31, n_estimators=80, random_state=random_state, n_jobs=-1, verbose=-1)
    t0 = time.time()
    clf_cap.fit(X_cap_tr, y_cap_tr)
    t_train_cap = time.time() - t0
    y_pred_cap = clf_cap.predict(X_cap_te)

    results.append({
        "Training & Test Regime": "1. Capped Stratified Dataset (Baseline)",
        "Classes": 34,
        "Flood Cap": "Capped at 25,000",
        "Accuracy": accuracy_score(y_cap_te, y_pred_cap),
        "Macro F1": f1_score(y_cap_te, y_pred_cap, average="macro", zero_division=0),
        "Weighted F1": f1_score(y_cap_te, y_pred_cap, average="weighted", zero_division=0),
        "Train Time (s)": round(t_train_cap, 2),
    })

    # -------------------------------------------------------------
    # Regime B: Un-Capped Natural Dataset (Floods Not Capped)
    # -------------------------------------------------------------
    X_uncap = df_uncapped[FEATURE_COLUMNS].values
    y_uncap = df_uncapped["label_idx"].values
    X_uncap_tr, X_uncap_te, y_uncap_tr, y_uncap_te = train_test_split(
        X_uncap, y_uncap, test_size=0.2, random_state=random_state
    )
    scaler_uncap = StandardScaler()
    X_uncap_tr = scaler_uncap.fit_transform(X_uncap_tr)
    X_uncap_te = scaler_uncap.transform(X_uncap_te)

    # 1. LightGBM on Un-Capped
    print("--> Evaluating LightGBM on Natural Un-Capped Stream (34 Classes)...", flush=True)
    clf_lgb_uncap = lgb.LGBMClassifier(num_leaves=63, n_estimators=100, learning_rate=0.1, random_state=random_state, n_jobs=-1, verbose=-1)
    t0 = time.time()
    clf_lgb_uncap.fit(X_uncap_tr, y_uncap_tr)
    t_train_lgb = time.time() - t0
    y_pred_lgb = clf_lgb_uncap.predict(X_uncap_te)

    results.append({
        "Training & Test Regime": "2. Natural Un-Capped Stream (LightGBM)",
        "Classes": 34,
        "Flood Cap": "NO CAP (Natural)",
        "Accuracy": accuracy_score(y_uncap_te, y_pred_lgb),
        "Macro F1": f1_score(y_uncap_te, y_pred_lgb, average="macro", zero_division=0),
        "Weighted F1": f1_score(y_uncap_te, y_pred_lgb, average="weighted", zero_division=0),
        "Train Time (s)": round(t_train_lgb, 2),
    })

    # 2. Random Forest on Un-Capped (The model used in Neto et al. Table 6)
    print("--> Evaluating Random Forest on Natural Un-Capped Stream (34 Classes)...", flush=True)
    clf_rf_uncap = RandomForestClassifier(n_estimators=60, max_depth=20, random_state=random_state, n_jobs=-1)
    t0 = time.time()
    clf_rf_uncap.fit(X_uncap_tr, y_uncap_tr)
    t_train_rf = time.time() - t0
    y_pred_rf = clf_rf_uncap.predict(X_uncap_te)

    results.append({
        "Training & Test Regime": "3. Natural Un-Capped Stream (Random Forest)",
        "Classes": 34,
        "Flood Cap": "NO CAP (Natural)",
        "Accuracy": accuracy_score(y_uncap_te, y_pred_rf),
        "Macro F1": f1_score(y_uncap_te, y_pred_rf, average="macro", zero_division=0),
        "Weighted F1": f1_score(y_uncap_te, y_pred_rf, average="weighted", zero_division=0),
        "Train Time (s)": round(t_train_rf, 2),
    })

    # 3. XGBoost Hist on Un-Capped (The model used in Almahaqeri et al.)
    print("--> Evaluating XGBoost Hist on Natural Un-Capped Stream (34 Classes)...", flush=True)
    clf_xgb_uncap = xgb.XGBClassifier(
        max_depth=8,
        n_estimators=80,
        learning_rate=0.1,
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
        eval_metric="mlogloss",
    )
    t0 = time.time()
    clf_xgb_uncap.fit(X_uncap_tr, y_uncap_tr)
    t_train_xgb = time.time() - t0
    y_pred_xgb = clf_xgb_uncap.predict(X_uncap_te)

    results.append({
        "Training & Test Regime": "4. Natural Un-Capped Stream (XGBoost Hist)",
        "Classes": 34,
        "Flood Cap": "NO CAP (Natural)",
        "Accuracy": accuracy_score(y_uncap_te, y_pred_xgb),
        "Macro F1": f1_score(y_uncap_te, y_pred_xgb, average="macro", zero_division=0),
        "Weighted F1": f1_score(y_uncap_te, y_pred_xgb, average="weighted", zero_division=0),
        "Train Time (s)": round(t_train_xgb, 2),
    })

    df_res = pd.DataFrame(results)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(RESULTS_PATH, index=False)

    print("\n" + "=" * 80)
    print(" EXPERIMENT 7 RESULTS SUMMARY &mdash; PURE 34 CLASSES (ZERO MERGING):")
    print("=" * 80)
    headers = ["Training & Test Regime", "Classes", "Flood Cap", "Accuracy", "Macro F1", "Weighted F1", "Train(s)"]
    table_data = [
        [
            r["Training & Test Regime"],
            r["Classes"],
            r["Flood Cap"],
            f"{r['Accuracy']*100:.2f}%",
            f"{r['Macro F1']:.4f}",
            f"{r['Weighted F1']:.4f}",
            f"{r['Train Time (s)']:.1f}",
        ]
        for r in results
    ]
    print(tabulate(table_data, headers=headers, tablefmt="github"))
    return df_res


if __name__ == "__main__":
    run_experiment_7()
