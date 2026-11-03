"""
Research Improvement 1: Loss and Class Weighting Calibration.
Investigates the impact of class weighting schemes on 34-class classification:
1. Linear Inverse ('balanced' - baseline)
2. Square-Root Dampened Inverse ('sqrt_balanced')
3. Log Dampened Inverse ('log_balanced')
4. Natural Unweighted ('unweighted')

Runs fast (~15-20 seconds) on a 100k stratified subset.
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
from sklearn.utils.class_weight import compute_class_weight
import lightgbm as lgb
from tabulate import tabulate

from src.config import FEATURE_COLUMNS, TARGET_COLUMN, THIRTY_FOUR_CLASSES, THIRTY_FOUR_TO_IDX

SAMPLE_PATH = BASE_DIR / "data" / "sample_stratified.csv"
RESULTS_PATH = BASE_DIR / "results" / "tables" / "exp1_weight_calibration.csv"

def compute_custom_weights(y_train, scheme="balanced"):
    classes = np.unique(y_train)
    n_samples = len(y_train)
    n_classes = len(classes)
    class_counts = pd.Series(y_train).value_counts().to_dict()
    
    if scheme == "unweighted":
        return None
    elif scheme == "balanced":
        # Standard inverse frequency
        weights = {c: n_samples / (n_classes * class_counts[c]) for c in classes}
        return weights
    elif scheme == "sqrt_balanced":
        # Square-root dampened inverse frequency
        raw_weights = {c: np.sqrt(n_samples / (n_classes * class_counts[c])) for c in classes}
        # Normalize so mean is 1.0
        mean_w = np.mean(list(raw_weights.values()))
        return {c: w / mean_w for c, w in raw_weights.items()}
    elif scheme == "log_balanced":
        # Log dampened inverse frequency
        raw_weights = {c: np.log1p(n_samples / (n_classes * class_counts[c])) for c in classes}
        mean_w = np.mean(list(raw_weights.values()))
        return {c: w / mean_w for c, w in raw_weights.items()}
    elif scheme == "clipped_balanced":
        # Inverse frequency clipped to max 10.0
        weights = {c: min(10.0, max(0.5, n_samples / (n_classes * class_counts[c]))) for c in classes}
        return weights
    else:
        raise ValueError(f"Unknown scheme: {scheme}")

def run_experiment_1(sample_size=100_000, random_state=42):
    print("=" * 80)
    print(" EXPERIMENT 1: CLASS WEIGHTING & LOSS CALIBRATION ABLATION")
    print("=" * 80)
    
    # 1. Load data
    print(f"[*] Loading data sample ({sample_size:,} stratified subset)...")
    df = pd.read_csv(SAMPLE_PATH)
    
    # Stratified subsampling if needed
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

    # Map labels to integers
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

    print(f"[*] Train set: {X_train.shape[0]:,} samples | Test set: {X_test.shape[0]:,} samples | 34 classes")

    schemes = [
        ("balanced (Baseline)", "balanced"),
        ("clipped_balanced (Cap=10)", "clipped_balanced"),
        ("sqrt_balanced (Dampened)", "sqrt_balanced"),
        ("log_balanced (Log Dampened)", "log_balanced"),
        ("unweighted (Natural Loss)", "unweighted"),
    ]

    results = []

    for name, scheme in schemes:
        print(f"\n--> Testing Weight Scheme: [{name}]...")
        weights_dict = compute_custom_weights(y_train, scheme=scheme)
        sample_weight = None
        if weights_dict is not None:
            sample_weight = np.array([weights_dict[cls] for cls in y_train])

        # LightGBM Classifier (fast and representative)
        clf = lgb.LGBMClassifier(
            n_estimators=80,
            learning_rate=0.08,
            num_leaves=31,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )

        t0 = time.time()
        clf.fit(X_train, y_train, sample_weight=sample_weight)
        train_time = time.time() - t0

        t0_inf = time.time()
        y_pred = clf.predict(X_test)
        inf_time_us = ((time.time() - t0_inf) / len(X_test)) * 1_000_000

        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)

        results.append({
            "Weighting Scheme": name,
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
    print(" EXPERIMENT 1 RESULTS SUMMARY:")
    print("=" * 80)
    headers = ["Scheme", "Accuracy", "Macro F1", "Weighted F1", "Macro Prec", "Macro Rec", "Train(s)", "Lat(us)"]
    table_data = [
        [
            r["Weighting Scheme"],
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
    run_experiment_1()
