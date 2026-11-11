"""
Strategy 2 Ablation Experiment Partitioner.
Purges every record belonging to an ambiguous feature vector (vectors carrying multiple conflicting labels)
from the datasets to create perfectly separable ablation partitions.
Primary Strategy 1 datasets remain completely untouched.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import src  # noqa: F401
    __package__ = "src"

from .config import (
    FEATURE_COLUMNS,
    ORIGINAL_SAMPLE_FILE,
    PARTITIONS_DIR,
    PARTITION_ASSIGNMENTS_FILE,
    TARGET_COLUMN,
)

ABLATION_DIR = PARTITIONS_DIR / "ablation_strategy2"
ABLATION_ASSIGNMENTS_FILE = ABLATION_DIR / "partition_assignments.csv"
ABLATION_METADATA_FILE = ABLATION_DIR / "metadata.json"


def generate_ablation_strategy2() -> Tuple[pd.DataFrame, Dict]:
    print("=" * 80)
    print("GENERATING STRATEGY 2 ABLATION PARTITIONS (PURGING AMBIGUOUS VECTORS)")
    print("=" * 80)

    ABLATION_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Strategy 1 Partition Assignments and Data
    df_assign = pd.read_csv(PARTITION_ASSIGNMENTS_FILE)
    df_raw = pd.read_csv(ORIGINAL_SAMPLE_FILE)

    df_clean = df_raw.iloc[df_assign["orig_idx"]].copy().reset_index(drop=True)
    df_clean["orig_idx"] = df_assign["orig_idx"].values
    df_clean["partition"] = df_assign["partition"].values
    df_clean["feat_hash"] = df_assign["feat_hash"].values

    total_clean_rows = len(df_clean)
    print(f"[*] Starting from Strategy 1 deduplicated rows: {total_clean_rows:,}")

    # 2. Identify all feature hashes that have >1 distinct label
    hash_label_counts = df_clean.groupby("feat_hash")[TARGET_COLUMN].nunique()
    ambiguous_hashes = set(hash_label_counts[hash_label_counts > 1].index)
    n_ambig_hashes = len(ambiguous_hashes)

    print(f"[*] Identified {n_ambig_hashes:,} unique ambiguous feature coordinates.")

    # 3. Purge all records belonging to ambiguous feature vectors
    ambig_mask = df_clean["feat_hash"].isin(ambiguous_hashes)
    n_purged_rows = int(ambig_mask.sum())
    df_separable = df_clean[~ambig_mask].copy().reset_index(drop=True)

    print(f"[*] Total ambiguous rows purged: {n_purged_rows:,} ({n_purged_rows / total_clean_rows * 100:.2f}%)")
    print(f"[*] Retained separable rows:    {len(df_separable):,} ({len(df_separable) / total_clean_rows * 100:.2f}%)")

    # 4. Partition counts in Strategy 2
    tr_sub = df_separable[df_separable["partition"] == "train"]
    va_sub = df_separable[df_separable["partition"] == "val"]
    te_sub = df_separable[df_separable["partition"] == "test"]

    print("\n--- Strategy 2 Ablation Partition Dimensions ---")
    print(f"  Train: {len(tr_sub):>8,} rows ({len(tr_sub)/len(df_separable)*100:5.2f}%) | Classes: {tr_sub[TARGET_COLUMN].nunique()} of 34")
    print(f"  Val:   {len(va_sub):>8,} rows ({len(va_sub)/len(df_separable)*100:5.2f}%) | Classes: {va_sub[TARGET_COLUMN].nunique()} of 34")
    print(f"  Test:  {len(te_sub):>8,} rows ({len(te_sub)/len(df_separable)*100:5.2f}%) | Classes: {te_sub[TARGET_COLUMN].nunique()} of 34")

    # 5. Exact Vector Independence Verification (Condition 2)
    print("\n--- Verifying Exact Feature-Vector Independence in Strategy 2 ---")
    tr_tuples = set(map(tuple, tr_sub[FEATURE_COLUMNS].to_numpy(dtype=np.float32)))
    va_tuples = set(map(tuple, va_sub[FEATURE_COLUMNS].to_numpy(dtype=np.float32)))
    te_tuples = set(map(tuple, te_sub[FEATURE_COLUMNS].to_numpy(dtype=np.float32)))

    eq_tr_va = len(tr_tuples.intersection(va_tuples))
    eq_tr_te = len(tr_tuples.intersection(te_tuples))
    eq_va_te = len(va_tuples.intersection(te_tuples))

    print(f"  Train & Val exact vector overlap:  {eq_tr_va}")
    print(f"  Train & Test exact vector overlap: {eq_tr_te}")
    print(f"  Val & Test exact vector overlap:   {eq_va_te}")

    assert eq_tr_va == 0, f"Ablation collision: Train & Val ({eq_tr_va})"
    assert eq_tr_te == 0, f"Ablation collision: Train & Test ({eq_tr_te})"
    assert eq_va_te == 0, f"Ablation collision: Val & Test ({eq_va_te})"
    print("  [PASSED] Strategy 2 partitions are strictly mutually disjoint (0 overlap).")

    # Verify that NO ambiguous vectors remain
    df_separable["hash_recheck"] = pd.util.hash_pandas_object(df_separable[FEATURE_COLUMNS], index=False)
    recheck_counts = df_separable.groupby("hash_recheck")[TARGET_COLUMN].nunique()
    remaining_ambig = (recheck_counts > 1).sum()
    assert remaining_ambig == 0, f"Failed: {remaining_ambig} ambiguous hashes still remain!"
    print(f"  [PASSED] 100% of remaining feature vectors map to exactly 1 class.")

    # 6. Save Strategy 2 Assignments & Metadata
    assignment_cols = ["orig_idx", "feat_hash", TARGET_COLUMN, "partition"]
    df_separable[assignment_cols].to_csv(ABLATION_ASSIGNMENTS_FILE, index=False)

    metadata = {
        "strategy": "Strategy 2: Ambiguous Feature Vectors Purged",
        "description": "Separate ablation partition isolating impact of empirical label collisions.",
        "original_deduplicated_rows": total_clean_rows,
        "ambiguous_feature_hashes_purged": n_ambig_hashes,
        "ambiguous_rows_purged": n_purged_rows,
        "purged_percentage": round(n_purged_rows / total_clean_rows * 100, 2),
        "separable_rows_total": len(df_separable),
        "partition_counts": {
            "train": len(tr_sub),
            "val": len(va_sub),
            "test": len(te_sub),
        },
        "classes_per_partition": {
            "train": int(tr_sub[TARGET_COLUMN].nunique()),
            "val": int(va_sub[TARGET_COLUMN].nunique()),
            "test": int(te_sub[TARGET_COLUMN].nunique()),
        },
        "exact_vector_overlaps": {
            "train_val": eq_tr_va,
            "train_test": eq_tr_te,
            "val_test": eq_va_te,
        },
        "per_class_counts": {
            k: int(v) for k, v in df_separable[TARGET_COLUMN].value_counts().items()
        },
    }

    with open(ABLATION_METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved Strategy 2 assignments to: {ABLATION_ASSIGNMENTS_FILE}")
    print(f"Saved Strategy 2 metadata to:    {ABLATION_METADATA_FILE}")
    print("=" * 80)
    return df_separable, metadata


if __name__ == "__main__":
    generate_ablation_strategy2()
