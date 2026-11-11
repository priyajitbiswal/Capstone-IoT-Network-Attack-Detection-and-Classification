"""
Research Improvement 4: Two-Stage Hierarchical (Cascaded) Classification.
Architecture:
- Stage 1: Fast 8-Category Macro Classifier (Benign, DDoS, DoS, Mirai, Recon, Spoofing, Web, BruteForce)
- Stage 2: Category-Specific Specialists (DDoS-12, DoS-4, Mirai-3, Recon-5, Spoofing-2, Web-6)
  * Categories with 1 class (Benign, BruteForce) bypass Stage 2 directly.

Benchmarks:
- Monolithic 34-Class Classifier (Baseline)
- Cascaded Two-Stage Architecture

Runs in ~30-40 seconds on the 100k stratified dataset.
"""

import sys
from pathlib import Path
import time
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import lightgbm as lgb
from tabulate import tabulate

from src.config import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    LABEL_MAPPING_8CLASSES,
    EIGHT_CLASSES,
    EIGHT_TO_IDX,
    IDX_TO_EIGHT,
    THIRTY_FOUR_CLASSES,
    THIRTY_FOUR_TO_IDX,
    IDX_TO_THIRTY_FOUR,
)

SAMPLE_PATH = BASE_DIR.parent / "data" / "processed" / "sample_stratified.csv"
RESULTS_PATH = BASE_DIR / "results" / "tables" / "exp4_hierarchical_cascade.csv"


def get_sqrt_balanced_weights(y_train):
    classes = np.unique(y_train)
    n_samples = len(y_train)
    n_classes = len(classes)
    class_counts = pd.Series(y_train).value_counts().to_dict()
    raw_weights = {c: np.sqrt(n_samples / (n_classes * class_counts[c])) for c in classes}
    mean_w = np.mean(list(raw_weights.values()))
    norm_weights = {c: w / mean_w for c, w in raw_weights.items()}
    return np.array([norm_weights[cls] for cls in y_train])


class HierarchicalCascadedClassifier:
    """
    Two-Stage Hierarchical Classifier:
    Stage 1: 8-Category Classifier
    Stage 2: Category Specialists
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.stage1_clf = None
        self.specialists = {}
        self.cat_to_subclasses = defaultdict(list)
        self.subclass_to_local_idx = {}
        self.local_idx_to_subclass = {}

    def fit(self, X_train, y_fine_train, y_cat_train):
        # 1. Train Stage 1 Macro Classifier
        sample_w_stage1 = get_sqrt_balanced_weights(y_cat_train)
        self.stage1_clf = lgb.LGBMClassifier(
            num_leaves=31,
            n_estimators=100,
            learning_rate=0.08,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,
        )
        self.stage1_clf.fit(X_train, y_cat_train, sample_weight=sample_w_stage1)

        # 2. Map fine-grained classes to categories
        for fine_idx in np.unique(y_fine_train):
            fine_name = IDX_TO_THIRTY_FOUR[fine_idx]
            cat_name = LABEL_MAPPING_8CLASSES[fine_name]
            cat_idx = EIGHT_TO_IDX[cat_name]
            self.cat_to_subclasses[cat_idx].append(fine_idx)

        # 3. Train Stage 2 Specialists for multi-class categories
        for cat_idx, fine_indices in self.cat_to_subclasses.items():
            if len(fine_indices) > 1:
                cat_mask = (y_cat_train == cat_idx)
                X_cat = X_train[cat_mask]
                y_fine_cat = y_fine_train[cat_mask]

                # Map to local class indices 0..K-1
                local_map = {orig: i for i, orig in enumerate(sorted(fine_indices))}
                reverse_local = {i: orig for orig, i in local_map.items()}
                self.subclass_to_local_idx[cat_idx] = local_map
                self.local_idx_to_subclass[cat_idx] = reverse_local

                y_local = np.array([local_map[cls] for cls in y_fine_cat])
                sample_w_local = get_sqrt_balanced_weights(y_local)

                spec_clf = lgb.LGBMClassifier(
                    num_leaves=31,
                    n_estimators=80,
                    learning_rate=0.08,
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbose=-1,
                )
                spec_clf.fit(X_cat, y_local, sample_weight=sample_w_local)
                self.specialists[cat_idx] = spec_clf
            else:
                # Single class category (e.g. Benign, BruteForce)
                self.specialists[cat_idx] = fine_indices[0]

        return self

    def predict(self, X_test):
        # Stage 1 Prediction
        stage1_preds = self.stage1_clf.predict(X_test)
        final_preds = np.zeros(len(X_test), dtype=int)

        # Stage 2 Specialist Routing
        for cat_idx in np.unique(stage1_preds):
            mask = (stage1_preds == cat_idx)
            if not np.any(mask):
                continue

            handler = self.specialists.get(cat_idx)
            if isinstance(handler, (int, np.integer)):
                # Directly map single subclass
                final_preds[mask] = handler
            elif handler is not None:
                # Specialist model inference
                X_sub = X_test[mask]
                local_preds = handler.predict(X_sub)
                reverse_map = self.local_idx_to_subclass[cat_idx]
                global_preds = np.array([reverse_map[lp] for lp in local_preds])
                final_preds[mask] = global_preds
            else:
                # Fallback if unseen
                final_preds[mask] = self.cat_to_subclasses[cat_idx][0]

        return final_preds, stage1_preds


def run_experiment_4(sample_size=100_000, random_state=42):
    print("=" * 80)
    print(" EXPERIMENT 4: TWO-STAGE HIERARCHICAL CASCADED CLASSIFIER")
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

    df["label_fine"] = df[TARGET_COLUMN].str.upper().map(THIRTY_FOUR_TO_IDX)
    df["label_cat_str"] = df[TARGET_COLUMN].str.upper().map(LABEL_MAPPING_8CLASSES)
    df["label_cat"] = df["label_cat_str"].map(EIGHT_TO_IDX)

    df = df.dropna(subset=["label_fine", "label_cat"])
    df["label_fine"] = df["label_fine"].astype(int)
    df["label_cat"] = df["label_cat"].astype(int)

    X = df[FEATURE_COLUMNS].values
    y_fine = df["label_fine"].values
    y_cat = df["label_cat"].values

    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=random_state, stratify=y_fine
    )

    X_train, X_test = X[train_idx], X[test_idx]
    y_fine_train, y_fine_test = y_fine[train_idx], y_fine[test_idx]
    y_cat_train, y_cat_test = y_cat[train_idx], y_cat[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    results = []

    # ----------------------------------------------------
    # Model 1: Monolithic 34-Class Baseline (LightGBM)
    # ----------------------------------------------------
    print("\n--> [1/2] Training Monolithic 34-Class Baseline...", flush=True)
    sample_w_mono = get_sqrt_balanced_weights(y_fine_train)
    t0_mono = time.time()
    mono_clf = lgb.LGBMClassifier(
        num_leaves=31,
        n_estimators=100,
        learning_rate=0.08,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )
    mono_clf.fit(X_train, y_fine_train, sample_weight=sample_w_mono)
    train_t_mono = time.time() - t0_mono

    t0_inf = time.time()
    y_pred_mono = mono_clf.predict(X_test)
    lat_mono = ((time.time() - t0_inf) / len(X_test)) * 1_000_000

    results.append({
        "Architecture": "Monolithic 34-Class Baseline",
        "Accuracy": accuracy_score(y_fine_test, y_pred_mono),
        "Macro F1": f1_score(y_fine_test, y_pred_mono, average="macro", zero_division=0),
        "Weighted F1": f1_score(y_fine_test, y_pred_mono, average="weighted", zero_division=0),
        "Macro Precision": precision_score(y_fine_test, y_pred_mono, average="macro", zero_division=0),
        "Macro Recall": recall_score(y_fine_test, y_pred_mono, average="macro", zero_division=0),
        "Train Time (s)": round(train_t_mono, 2),
        "Latency (us/flow)": round(lat_mono, 2),
    })

    # ----------------------------------------------------
    # Model 2: Two-Stage Hierarchical Cascaded Architecture
    # ----------------------------------------------------
    print("--> [2/2] Training Two-Stage Hierarchical Cascade...", flush=True)
    t0_casc = time.time()
    cascade = HierarchicalCascadedClassifier(random_state=random_state)
    cascade.fit(X_train, y_fine_train, y_cat_train)
    train_t_casc = time.time() - t0_casc

    t0_inf = time.time()
    y_pred_casc, y_stage1_preds = cascade.predict(X_test)
    lat_casc = ((time.time() - t0_inf) / len(X_test)) * 1_000_000

    stage1_acc = accuracy_score(y_cat_test, y_stage1_preds)
    print(f"    [*] Stage 1 (8-Category) Accuracy: {stage1_acc*100:.2f}%")

    results.append({
        "Architecture": "Two-Stage Hierarchical Cascade",
        "Accuracy": accuracy_score(y_fine_test, y_pred_casc),
        "Macro F1": f1_score(y_fine_test, y_pred_casc, average="macro", zero_division=0),
        "Weighted F1": f1_score(y_fine_test, y_pred_casc, average="weighted", zero_division=0),
        "Macro Precision": precision_score(y_fine_test, y_pred_casc, average="macro", zero_division=0),
        "Macro Recall": recall_score(y_fine_test, y_pred_casc, average="macro", zero_division=0),
        "Train Time (s)": round(train_t_casc, 2),
        "Latency (us/flow)": round(lat_casc, 2),
    })

    df_res = pd.DataFrame(results)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(RESULTS_PATH, index=False)

    print("\n" + "=" * 80)
    print(" EXPERIMENT 4 RESULTS SUMMARY:")
    print("=" * 80)
    headers = ["Architecture", "Accuracy", "Macro F1", "Weighted F1", "Macro Prec", "Macro Rec", "Train(s)", "Lat(us)"]
    table_data = [
        [
            r["Architecture"],
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
    run_experiment_4()
