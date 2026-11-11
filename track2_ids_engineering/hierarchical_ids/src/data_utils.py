"""
Data loading, taxonomy mappings, and strictly leak-free partition helpers for
Track 2 Hierarchical IDS evaluation (Architectures E1-E4).

Reuses Track 1's verified SHA-256 group-hash partitions and serialized
`preprocessor_strategy1.joblib` read-only so all architectures evaluate on
identical features, preprocessing, and populations.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd

import track1_dataset_evaluation.src as _t1_compat  # noqa: F401 (registers src_v2 pickle aliases)
from track1_dataset_evaluation.src.config import (
    BINARY_CLASSES,
    BINARY_TO_IDX,
    EIGHT_CLASSES,
    EIGHT_TO_IDX,
    FEATURE_COLUMNS,
    FUNCTIONAL_THREAT_TIERS,
    LABEL_MAPPING_8CLASSES,
    LABEL_MAPPING_BINARY,
    MODELS_V2_DIR as TRACK1_MODELS_DIR,
    ORIGINAL_SAMPLE_FILE,
    PARTITIONS_DIR as TRACK1_PARTITIONS_DIR,
    PARTITION_ASSIGNMENTS_FILE,
    RANDOM_SEED,
    TARGET_COLUMN,
    THIRTY_FOUR_CLASSES,
    THIRTY_FOUR_TO_IDX,
)
from track1_dataset_evaluation.src.preprocessing import encode_target_labels

# Track 2 Hierarchical IDS Directory Anchor
HIER_ROOT = Path(__file__).resolve().parent.parent
HIER_MODELS_DIR = HIER_ROOT / "models"
HIER_RESULTS_DIR = HIER_ROOT / "results"
HIER_DOCS_DIR = HIER_ROOT / "docs"

# Canonical Flood vs. Non-Flood Split (matches Track 1 Tier 1 Volumetric Floods = DDoS + DoS)
FLOOD_CATEGORIES = {"DDoS", "DoS"}
NON_FLOOD_CATEGORIES = {"Mirai", "Recon", "Spoofing", "Web", "BruteForce"}
ATTACK_CATEGORIES_7 = [c for c in EIGHT_CLASSES if c != "Benign"]
ATTACK_CAT7_TO_IDX = {c: i for i, c in enumerate(ATTACK_CATEGORIES_7)}

FLOOD_CLASSES = sorted(
    [lbl for lbl, cat in LABEL_MAPPING_8CLASSES.items() if cat in FLOOD_CATEGORIES]
)
NON_FLOOD_CLASSES = sorted(
    [lbl for lbl, cat in LABEL_MAPPING_8CLASSES.items() if cat in NON_FLOOD_CATEGORIES]
)

FLOOD_GLOBAL_INDICES = [THIRTY_FOUR_TO_IDX[c] for c in FLOOD_CLASSES]
NON_FLOOD_GLOBAL_INDICES = [THIRTY_FOUR_TO_IDX[c] for c in NON_FLOOD_CLASSES]
BENIGN_GLOBAL_IDX = THIRTY_FOUR_TO_IDX["BENIGN"]

# Local <-> Global mappings for specialists
FLOOD_LOCAL_TO_GLOBAL = {i: THIRTY_FOUR_TO_IDX[c] for i, c in enumerate(FLOOD_CLASSES)}
FLOOD_GLOBAL_TO_LOCAL = {g: i for i, g in FLOOD_LOCAL_TO_GLOBAL.items()}

NON_FLOOD_LOCAL_TO_GLOBAL = {i: THIRTY_FOUR_TO_IDX[c] for i, c in enumerate(NON_FLOOD_CLASSES)}
NON_FLOOD_GLOBAL_TO_LOCAL = {g: i for i, g in NON_FLOOD_LOCAL_TO_GLOBAL.items()}

# Rare / Minority attack classes (< 5,000 samples in stratified working set)
MINORITY_ATTACK_CLASSES = [
    "UPLOADING_ATTACK",
    "RECON-PINGSWEEP",
    "BACKDOOR_MALWARE",
    "XSS",
    "COMMANDINJECTION",
    "SQLINJECTION",
    "BROWSERHIJACKING",
    "DICTIONARYBRUTEFORCE",
]

# Pre-validate taxonomy completeness at import time
assert len(FLOOD_CLASSES) == 16, f"Expected 16 flood classes, got {len(FLOOD_CLASSES)}"
assert len(NON_FLOOD_CLASSES) == 17, f"Expected 17 non-flood classes, got {len(NON_FLOOD_CLASSES)}"
assert set(FLOOD_CLASSES).isdisjoint(set(NON_FLOOD_CLASSES))
assert set(FLOOD_CLASSES) | set(NON_FLOOD_CLASSES) | {"BENIGN"} == set(THIRTY_FOUR_CLASSES)
assert set(FLOOD_CLASSES) == set(FUNCTIONAL_THREAT_TIERS["Tier 1: Volumetric Floods"])


def load_strategy1_preprocessor():
    """Loads Track 1's strictly training-fitted Strategy 1 LeakFreePreprocessor read-only."""
    prep_path = TRACK1_MODELS_DIR / "preprocessor_strategy1.joblib"
    if not prep_path.exists():
        raise FileNotFoundError(f"Missing Track 1 preprocessor at {prep_path}")
    return joblib.load(prep_path)


def encode_all_granularities(series: pd.Series) -> Dict[str, np.ndarray]:
    """Encodes raw Label Series into binary, 8-class category, 34-class fine, and flood masks."""
    y_bin, _ = encode_target_labels(series, "binary")
    y_cat, _ = encode_target_labels(series, "category")
    y_fine, _ = encode_target_labels(series, "fine_grained")

    raw_upper = series.astype(str).str.strip().str.upper()
    cat_str = raw_upper.map(LABEL_MAPPING_8CLASSES)
    is_attack = y_bin == BINARY_TO_IDX["Attack"]
    is_flood = cat_str.isin(FLOOD_CATEGORIES).to_numpy(dtype=bool)
    is_non_flood = cat_str.isin(NON_FLOOD_CATEGORIES).to_numpy(dtype=bool)

    return {
        "y_binary": y_bin,
        "y_category": y_cat,
        "y_fine": y_fine,
        "is_attack": is_attack,
        "is_flood": is_flood,
        "is_non_flood": is_non_flood,
    }


def load_track1_splits(
    include_test: bool = False,
    include_cross_capture: bool = False,
    verify_disjoint: bool = True,
) -> Dict[str, Dict[str, object]]:
    """
    Loads Track 1's Strategy 1 leak-free splits and applies `preprocessor_strategy1.joblib`
    strictly read-only.

    Test and cross-capture splits are ONLY loaded when explicitly requested for final evaluation.
    """
    if not PARTITION_ASSIGNMENTS_FILE.exists():
        raise FileNotFoundError(f"Missing partition assignments: {PARTITION_ASSIGNMENTS_FILE}")
    if not ORIGINAL_SAMPLE_FILE.exists():
        raise FileNotFoundError(f"Missing stratified sample: {ORIGINAL_SAMPLE_FILE}")

    prep = load_strategy1_preprocessor()
    df_assign = pd.read_csv(PARTITION_ASSIGNMENTS_FILE)
    df_raw = pd.read_csv(ORIGINAL_SAMPLE_FILE)
    df_clean = df_raw.iloc[df_assign["orig_idx"]].copy().reset_index(drop=True)
    df_clean["partition"] = df_assign["partition"].values

    splits: Dict[str, Dict[str, object]] = {}

    def _pack(df_sub: pd.DataFrame) -> Dict[str, object]:
        X_clean = prep.clean_transform(df_sub)
        X_scaled = prep.transform_scaler(X_clean)
        labels = encode_all_granularities(df_sub[TARGET_COLUMN])
        return {
            "X": X_scaled,
            "raw_df_len": len(df_sub),
            **labels,
        }

    train_df = df_clean[df_clean["partition"] == "train"].reset_index(drop=True)
    val_df = df_clean[df_clean["partition"] == "val"].reset_index(drop=True)
    assert len(train_df) == 350821, f"Unexpected train size {len(train_df)}"
    assert len(val_df) == 75172, f"Unexpected val size {len(val_df)}"

    splits["train"] = _pack(train_df)
    splits["val"] = _pack(val_df)

    hash_sets = {}
    if verify_disjoint:
        hash_sets["train"] = set(pd.util.hash_pandas_object(train_df[FEATURE_COLUMNS], index=False))
        hash_sets["val"] = set(pd.util.hash_pandas_object(val_df[FEATURE_COLUMNS], index=False))
        assert len(hash_sets["train"] & hash_sets["val"]) == 0, "Train/Val feature overlap!"

    if include_test:
        test_df = df_clean[df_clean["partition"] == "test"].reset_index(drop=True)
        assert len(test_df) == 75105, f"Unexpected internal test size {len(test_df)}"
        splits["test"] = _pack(test_df)
        if verify_disjoint:
            hash_sets["test"] = set(pd.util.hash_pandas_object(test_df[FEATURE_COLUMNS], index=False))
            assert len(hash_sets["train"] & hash_sets["test"]) == 0, "Train/Test feature overlap!"
            assert len(hash_sets["val"] & hash_sets["test"]) == 0, "Val/Test feature overlap!"

    if include_cross_capture:
        cross_path = TRACK1_PARTITIONS_DIR / "cross_capture_test_clean.csv"
        if not cross_path.exists():
            raise FileNotFoundError(f"Missing cross-capture test set: {cross_path}")
        cross_df = pd.read_csv(cross_path)
        assert len(cross_df) == 57280, f"Unexpected cross-capture size {len(cross_df)}"
        splits["cross"] = _pack(cross_df)
        if verify_disjoint:
            hash_sets["cross"] = set(pd.util.hash_pandas_object(cross_df[FEATURE_COLUMNS], index=False))
            for k in list(hash_sets.keys()):
                if k != "cross":
                    assert len(hash_sets[k] & hash_sets["cross"]) == 0, f"{k}/Cross feature overlap!"

    return splits


def compute_sample_weights(
    y: np.ndarray,
    scheme: str = "balanced",
    clip_cap: float = 10.0,
) -> Optional[np.ndarray]:
    """
    Computes per-sample weights for class-aware training.
    Supported schemes: 'unweighted', 'balanced', 'sqrt_balanced', 'log_balanced', 'clipped_balanced'.
    """
    if scheme == "unweighted" or scheme is None:
        return None

    classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(classes)
    count_map = dict(zip(classes, counts))

    if scheme == "balanced":
        raw_w = {c: n_samples / (n_classes * count_map[c]) for c in classes}
    elif scheme == "sqrt_balanced":
        raw_w = {c: np.sqrt(n_samples / (n_classes * count_map[c])) for c in classes}
    elif scheme == "log_balanced":
        raw_w = {c: np.log1p(n_samples / (n_classes * count_map[c])) for c in classes}
    elif scheme == "clipped_balanced":
        raw_w = {c: min(clip_cap, n_samples / (n_classes * count_map[c])) for c in classes}
    else:
        raise ValueError(f"Unknown weighting scheme: {scheme}")

    mean_w = float(np.mean(list(raw_w.values())))
    norm_w = {c: w / mean_w for c, w in raw_w.items()}
    return np.asarray([norm_w[cls] for cls in y], dtype=np.float32)
