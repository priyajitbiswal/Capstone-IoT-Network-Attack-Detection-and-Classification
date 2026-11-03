"""
Strict four-partition exact-vector equality and hash verification script.
Verifies pairwise disjointness across:
  1. Training Partition
  2. Validation Partition
  3. Internal Test Partition
  4. Clean Cross-Capture Holdout Partition
Exits with code 0 on verified success, or raises AssertionError with non-zero exit code on failure.
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

from src_v2.config import (
    FEATURE_COLUMNS,
    ORIGINAL_SAMPLE_FILE,
    PARTITIONS_DIR,
    PARTITION_ASSIGNMENTS_FILE,
)

CROSS_CAPTURE_CLEAN_FILE = PARTITIONS_DIR / "cross_capture_test_clean.csv"


def verify_all_partitions() -> int:
    print("=" * 80)
    print("FOUR-PARTITION INDEPENDENCE & EXACT FEATURE-VECTOR VERIFICATION")
    print("=" * 80)

    # 1. Load Primary Strategy 1 Partition Assignments
    if not PARTITION_ASSIGNMENTS_FILE.exists():
        raise FileNotFoundError(f"Missing {PARTITION_ASSIGNMENTS_FILE}")
    if not ORIGINAL_SAMPLE_FILE.exists():
        raise FileNotFoundError(f"Missing {ORIGINAL_SAMPLE_FILE}")
    if not CROSS_CAPTURE_CLEAN_FILE.exists():
        raise FileNotFoundError(f"Missing {CROSS_CAPTURE_CLEAN_FILE}")

    print("[*] Loading internal partitions from sample_stratified.csv...")
    df_assign = pd.read_csv(PARTITION_ASSIGNMENTS_FILE)
    df_raw = pd.read_csv(ORIGINAL_SAMPLE_FILE)

    df_clean = df_raw.iloc[df_assign["orig_idx"]].copy().reset_index(drop=True)
    df_clean["partition"] = df_assign["partition"].values

    train_df = df_clean[df_clean["partition"] == "train"].copy().reset_index(drop=True)
    val_df = df_clean[df_clean["partition"] == "val"].copy().reset_index(drop=True)
    test_df = df_clean[df_clean["partition"] == "test"].copy().reset_index(drop=True)

    print("[*] Loading clean cross-capture holdout...")
    cross_df = pd.read_csv(CROSS_CAPTURE_CLEAN_FILE)

    print("\n--- Partition Dimensions & Classes ---")
    print(f"  1. Train Partition:         {len(train_df):>8,} rows | {train_df['Label'].nunique()} of 34 classes")
    print(f"  2. Validation Partition:    {len(val_df):>8,} rows | {val_df['Label'].nunique()} of 34 classes")
    print(f"  3. Internal Test Partition: {len(test_df):>8,} rows | {test_df['Label'].nunique()} of 34 classes")
    print(f"  4. Cross-Capture Holdout:   {len(cross_df):>8,} rows | {cross_df['Label'].nunique()} of 34 classes")

    # 2. Hash-based Disjointness Check (64-bit MurmurHash3)
    print("\n--- Check 1: 64-bit Feature Hash Disjointness ---")
    train_h = set(pd.util.hash_pandas_object(train_df[FEATURE_COLUMNS], index=False))
    val_h = set(pd.util.hash_pandas_object(val_df[FEATURE_COLUMNS], index=False))
    test_h = set(pd.util.hash_pandas_object(test_df[FEATURE_COLUMNS], index=False))
    cross_h = set(pd.util.hash_pandas_object(cross_df[FEATURE_COLUMNS], index=False))

    h_tr_va = len(train_h.intersection(val_h))
    h_tr_te = len(train_h.intersection(test_h))
    h_va_te = len(val_h.intersection(test_h))
    h_tr_cr = len(train_h.intersection(cross_h))
    h_va_cr = len(val_h.intersection(cross_h))
    h_te_cr = len(test_h.intersection(cross_h))

    print(f"  Train & Val hash overlap:           {h_tr_va}")
    print(f"  Train & Internal Test hash overlap: {h_tr_te}")
    print(f"  Val & Internal Test hash overlap:   {h_va_te}")
    print(f"  Train & Cross-Capture hash overlap: {h_tr_cr}")
    print(f"  Val & Cross-Capture hash overlap:   {h_va_cr}")
    print(f"  Test & Cross-Capture hash overlap:  {h_te_cr}")

    assert h_tr_va == 0, f"Hash collision: Train & Val ({h_tr_va})"
    assert h_tr_te == 0, f"Hash collision: Train & Internal Test ({h_tr_te})"
    assert h_va_te == 0, f"Hash collision: Val & Internal Test ({h_va_te})"
    assert h_tr_cr == 0, f"Hash collision: Train & Cross-Capture ({h_tr_cr})"
    assert h_va_cr == 0, f"Hash collision: Val & Cross-Capture ({h_va_cr})"
    assert h_te_cr == 0, f"Hash collision: Test & Cross-Capture ({h_te_cr})"
    print("  [PASSED] All 6 pairwise hash intersections are strictly 0.")

    # 3. Exact Float-Vector Equality Check (Condition 5)
    print("\n--- Check 2: Actual Feature-Vector Equality (Exact Float Tuples) ---")
    train_tuples = set(map(tuple, train_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)))
    val_tuples = set(map(tuple, val_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)))
    test_tuples = set(map(tuple, test_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)))
    cross_tuples = set(map(tuple, cross_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)))

    print(f"  Train unique vectors:         {len(train_tuples):>8,}")
    print(f"  Val unique vectors:           {len(val_tuples):>8,}")
    print(f"  Internal Test unique vectors: {len(test_tuples):>8,}")
    print(f"  Cross-Capture unique vectors: {len(cross_tuples):>8,}")

    eq_tr_va = len(train_tuples.intersection(val_tuples))
    eq_tr_te = len(train_tuples.intersection(test_tuples))
    eq_va_te = len(val_tuples.intersection(test_tuples))
    eq_tr_cr = len(train_tuples.intersection(cross_tuples))
    eq_va_cr = len(val_tuples.intersection(cross_tuples))
    eq_te_cr = len(test_tuples.intersection(cross_tuples))

    print("\n  Pairwise Exact Tuple Overlaps:")
    print(f"    Train & Val:           {eq_tr_va}")
    print(f"    Train & Internal Test: {eq_tr_te}")
    print(f"    Val & Internal Test:   {eq_va_te}")
    print(f"    Train & Cross-Capture: {eq_tr_cr}")
    print(f"    Val & Cross-Capture:   {eq_va_cr}")
    print(f"    Test & Cross-Capture:  {eq_te_cr}")

    assert eq_tr_va == 0, f"Vector collision: Train & Val ({eq_tr_va})"
    assert eq_tr_te == 0, f"Vector collision: Train & Internal Test ({eq_tr_te})"
    assert eq_va_te == 0, f"Vector collision: Val & Internal Test ({eq_va_te})"
    assert eq_tr_cr == 0, f"Vector collision: Train & Cross-Capture ({eq_tr_cr})"
    assert eq_va_cr == 0, f"Vector collision: Val & Cross-Capture ({eq_va_cr})"
    assert eq_te_cr == 0, f"Vector collision: Test & Cross-Capture ({eq_te_cr})"
    print("  [PASSED] All 6 pairwise exact tuple intersections are strictly 0.")

    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETE: Zero duplicate feature-vector overlap verified.")
    print("All four partitions are strictly mutually disjoint.")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(verify_all_partitions())

