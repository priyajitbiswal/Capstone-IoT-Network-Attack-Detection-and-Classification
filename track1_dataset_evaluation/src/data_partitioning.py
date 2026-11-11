"""
Module for reproducible, leak-free dataset partitioning and duplicate investigation.
Creates a verified 70/15/15 Train/Validation/Test partition with strict zero feature overlap.
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
    PARTITION_METADATA_FILE,
    RANDOM_SEED,
    TARGET_COLUMN,
)


def investigate_duplicates(df: pd.DataFrame) -> Dict:
    """
    Detailed audit of all duplicate categories in the working dataset.
    """
    features = FEATURE_COLUMNS
    total_rows = len(df)

    # 1. Exact duplicate rows (features + label identical)
    exact_dup_mask = df.duplicated(subset=features + [TARGET_COLUMN], keep='first')
    exact_dup_all_mask = df.duplicated(subset=features + [TARGET_COLUMN], keep=False)

    # 2. Duplicate feature vectors (ignoring label)
    feat_dup_mask = df.duplicated(subset=features, keep='first')
    feat_dup_all_mask = df.duplicated(subset=features, keep=False)

    # 3. Conflicting labels (same feature vector, multiple distinct labels)
    feat_hashes = pd.util.hash_pandas_object(df[features], index=False)
    df_temp = pd.DataFrame({"hash": feat_hashes, "label": df[TARGET_COLUMN]})
    grouped_labels = df_temp.groupby("hash")["label"].nunique()
    conflicting_hashes = set(grouped_labels[grouped_labels > 1].index)
    conflicting_rows_mask = feat_hashes.isin(conflicting_hashes)

    stats = {
        "total_records": total_rows,
        "exact_duplicates": {
            "redundant_rows": int(exact_dup_mask.sum()),
            "redundant_pct": float(round(exact_dup_mask.sum() / total_rows * 100, 3)),
            "total_rows_in_exact_clusters": int(exact_dup_all_mask.sum()),
        },
        "feature_vector_duplicates": {
            "redundant_vectors": int(feat_dup_mask.sum()),
            "redundant_pct": float(round(feat_dup_mask.sum() / total_rows * 100, 3)),
            "total_rows_in_feature_clusters": int(feat_dup_all_mask.sum()),
        },
        "conflicting_label_collisions": {
            "unique_conflicting_hashes": int(len(conflicting_hashes)),
            "total_conflicting_rows": int(conflicting_rows_mask.sum()),
            "conflicting_pct": float(round(conflicting_rows_mask.sum() / total_rows * 100, 3)),
        },
    }
    return stats


def generate_leak_free_partitions(
    sample_path: Path = ORIGINAL_SAMPLE_FILE,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Constructs a deterministic, leak-free 70/15/15 partition:
      1. Deduplicates exact duplicate rows (preserving 1 unique instance per flow).
      2. Groups identical feature vectors together so no feature vector spans multiple partitions.
      3. Verifies exactly 0.00% feature overlap across train, val, and test.
      4. Saves partition assignments and metadata without modifying the original dataset.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, "Ratios must sum to 1.0"

    print(f"Loading original sample from: {sample_path}")
    df = pd.read_csv(sample_path)
    df["orig_idx"] = df.index

    # 1. Investigate original duplicates
    dup_stats = investigate_duplicates(df)

    # 2. Deduplicate exact duplicate rows
    exact_dup_mask = df.duplicated(subset=FEATURE_COLUMNS + [TARGET_COLUMN], keep="first")
    df_clean = df[~exact_dup_mask].copy().reset_index(drop=True)
    n_removed_exact = int(exact_dup_mask.sum())

    # 3. Compute deterministic feature hashes
    df_clean["feat_hash"] = pd.util.hash_pandas_object(df_clean[FEATURE_COLUMNS], index=False)

    # Unique feature vectors
    unique_hashes = df_clean["feat_hash"].unique()
    np.random.seed(random_seed)
    shuffled_hashes = np.random.permutation(unique_hashes)

    n_total = len(shuffled_hashes)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_hash_set = set(shuffled_hashes[:n_train])
    val_hash_set = set(shuffled_hashes[n_train:n_train + n_val])
    test_hash_set = set(shuffled_hashes[n_train + n_val:])

    # Assign partition
    df_clean["partition"] = "none"
    df_clean.loc[df_clean["feat_hash"].isin(train_hash_set), "partition"] = "train"
    df_clean.loc[df_clean["feat_hash"].isin(val_hash_set), "partition"] = "val"
    df_clean.loc[df_clean["feat_hash"].isin(test_hash_set), "partition"] = "test"

    # 4. Strict Overlap Verification
    train_hashes = set(df_clean.loc[df_clean["partition"] == "train", "feat_hash"])
    val_hashes = set(df_clean.loc[df_clean["partition"] == "val", "feat_hash"])
    test_hashes = set(df_clean.loc[df_clean["partition"] == "test", "feat_hash"])

    ov_train_val = len(train_hashes.intersection(val_hashes))
    ov_train_test = len(train_hashes.intersection(test_hashes))
    ov_val_test = len(val_hashes.intersection(test_hashes))

    assert ov_train_val == 0, f"Leakage detected between train and val: {ov_train_val} overlap"
    assert ov_train_test == 0, f"Leakage detected between train and test: {ov_train_test} overlap"
    assert ov_val_test == 0, f"Leakage detected between val and test: {ov_val_test} overlap"

    # Partition sizes and class counts
    counts_by_partition = df_clean["partition"].value_counts().to_dict()
    classes_by_partition = {
        p: int(df_clean.loc[df_clean["partition"] == p, TARGET_COLUMN].nunique())
        for p in ["train", "val", "test"]
    }

    # Per-class distribution across partitions
    per_class_dist = {}
    for cls_name, grp in df_clean.groupby(TARGET_COLUMN):
        p_counts = grp["partition"].value_counts().to_dict()
        per_class_dist[str(cls_name)] = {
            "total": len(grp),
            "train": p_counts.get("train", 0),
            "val": p_counts.get("val", 0),
            "test": p_counts.get("test", 0),
        }

    # Build metadata dictionary
    metadata = {
        "source_dataset": str(sample_path.resolve()),
        "random_seed": random_seed,
        "train_ratio_target": train_ratio,
        "val_ratio_target": val_ratio,
        "test_ratio_target": test_ratio,
        "duplicate_investigation": dup_stats,
        "exact_duplicates_removed": n_removed_exact,
        "clean_rows_total": len(df_clean),
        "unique_feature_vectors": len(unique_hashes),
        "actual_partition_counts": {
            "train": counts_by_partition.get("train", 0),
            "val": counts_by_partition.get("val", 0),
            "test": counts_by_partition.get("test", 0),
        },
        "actual_partition_percentages": {
            "train": round(counts_by_partition.get("train", 0) / len(df_clean) * 100, 2),
            "val": round(counts_by_partition.get("val", 0) / len(df_clean) * 100, 2),
            "test": round(counts_by_partition.get("test", 0) / len(df_clean) * 100, 2),
        },
        "classes_represented": classes_by_partition,
        "feature_leakage_overlap_counts": {
            "train_val": ov_train_val,
            "train_test": ov_train_test,
            "val_test": ov_val_test,
        },
        "per_class_distribution": per_class_dist,
    }

    # Save partition assignments (orig_idx, feat_hash, label, partition)
    PARTITIONS_DIR.mkdir(parents=True, exist_ok=True)
    assignment_cols = ["orig_idx", "feat_hash", TARGET_COLUMN, "partition"]
    df_clean[assignment_cols].to_csv(PARTITION_ASSIGNMENTS_FILE, index=False)
    print(f"Saved partition assignments to: {PARTITION_ASSIGNMENTS_FILE}")

    # Save partition metadata
    with open(PARTITION_METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved partition metadata to: {PARTITION_METADATA_FILE}")

    return df_clean, metadata


if __name__ == "__main__":
    generate_leak_free_partitions()

