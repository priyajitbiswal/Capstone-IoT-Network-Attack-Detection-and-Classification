"""
Corrected, leak-free preprocessing pipeline (Version 2).
Ensures all learned transformations (sanitization parameters and StandardScaler)
are fitted strictly on the training partition only.
"""

from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import (
    BINARY_CLASSES,
    BINARY_TO_IDX,
    EIGHT_CLASSES,
    EIGHT_TO_IDX,
    FEATURE_COLUMNS,
    LABEL_MAPPING_8CLASSES,
    LABEL_MAPPING_BINARY,
    MODELS_V2_DIR,
    ORIGINAL_SAMPLE_FILE,
    PARTITION_ASSIGNMENTS_FILE,
    TARGET_COLUMN,
    THIRTY_FOUR_CLASSES,
    THIRTY_FOUR_TO_IDX,
)


class LeakFreePreprocessor:
    """
    Stateful preprocessor that learns feature cleaning thresholds
    and standardization statistics strictly on training data.
    """

    def __init__(self, feature_cols: Optional[List[str]] = None):
        self.feature_cols = feature_cols if feature_cols is not None else FEATURE_COLUMNS
        self.column_finite_max: Dict[str, float] = {}
        self.scaler: Optional[StandardScaler] = None
        self.is_fitted: bool = False

    def fit(self, df_train: pd.DataFrame) -> "LeakFreePreprocessor":
        """
        Learns finite maxima strictly from training partition.
        """
        self.column_finite_max = {}
        for col in self.feature_cols:
            col_vals = df_train[col].to_numpy(dtype=np.float64, copy=False)
            inf_mask = np.isinf(col_vals)
            nan_mask = np.isnan(col_vals)
            finite_mask = ~(inf_mask | nan_mask)

            if finite_mask.any():
                self.column_finite_max[col] = float(col_vals[finite_mask].max())
            else:
                self.column_finite_max[col] = 0.0

        self.is_fitted = True
        return self

    def clean_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Applies learned cleaning thresholds to any partition:
          - Replaces +inf with training finite max
          - Replaces -inf with 0.0
          - Replaces NaN with 0.0
          - Enforces float32 precision
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted on training data before transforming.")

        cleaned_cols = []
        for col in self.feature_cols:
            vals = df[col].to_numpy(dtype=np.float32, copy=True)
            # Handle +inf
            pos_inf = np.isposinf(vals)
            if pos_inf.any():
                vals[pos_inf] = self.column_finite_max.get(col, 0.0)
            # Handle -inf
            neg_inf = np.isneginf(vals)
            if neg_inf.any():
                vals[neg_inf] = 0.0
            # Handle NaN
            nan_mask = np.isnan(vals)
            if nan_mask.any():
                vals[nan_mask] = 0.0

            cleaned_cols.append(vals)

        return np.column_stack(cleaned_cols).astype(np.float32)

    def fit_transform_scaler(self, X_train: np.ndarray) -> np.ndarray:
        """
        Fits StandardScaler strictly on cleaned X_train.
        """
        self.scaler = StandardScaler()
        return self.scaler.fit_transform(X_train).astype(np.float32)

    def transform_scaler(self, X: np.ndarray) -> np.ndarray:
        """
        Transforms evaluation data using scaler fitted on training data.
        """
        if self.scaler is None:
            raise RuntimeError("StandardScaler has not been fitted.")
        return self.scaler.transform(X).astype(np.float32)


def encode_target_labels(
    series: pd.Series,
    target_type: Literal["binary", "category", "fine_grained"] = "binary",
) -> Tuple[np.ndarray, List[str]]:
    """
    Encodes raw target labels into zero-indexed integers according to target granularity.
    """
    raw_labels = series.astype(str).str.strip().str.upper()

    if target_type == "binary":
        mapped = raw_labels.map(LABEL_MAPPING_BINARY).fillna("Attack")
        y = mapped.map(BINARY_TO_IDX).values.astype(np.int64)
        return y, BINARY_CLASSES

    elif target_type == "category":
        mapped = raw_labels.map(LABEL_MAPPING_8CLASSES).fillna("DDoS")
        y = mapped.map(EIGHT_TO_IDX).values.astype(np.int64)
        return y, EIGHT_CLASSES

    elif target_type == "fine_grained":
        mapped = raw_labels.map(THIRTY_FOUR_TO_IDX)
        # Fallback to 0 if unknown
        y = mapped.fillna(0).values.astype(np.int64)
        return y, THIRTY_FOUR_CLASSES

    else:
        raise ValueError(f"Unknown target_type: {target_type}")


def load_partitioned_datasets(
    target_type: Literal["binary", "category", "fine_grained"] = "binary",
    scale_features: bool = True,
    sample_file: Path = ORIGINAL_SAMPLE_FILE,
    assignments_file: Path = PARTITION_ASSIGNMENTS_FILE,
    save_preprocessor_path: Optional[Path] = None,
    load_test: bool = False,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],           # (X_train, y_train)
    Tuple[np.ndarray, np.ndarray],           # (X_val, y_val)
    Tuple[Optional[np.ndarray], Optional[np.ndarray]],  # (X_test, y_test)
    List[str],                               # class_names
    LeakFreePreprocessor,                   # fitted preprocessor
]:
    """
    Loads partitioned dataset according to verified partition assignments.
    Fits all preprocessing strictly on train, and transforms val.
    Test partition is strictly sealed and untouched unless load_test=True.
    
    Returns:
        (X_train, y_train), (X_val, y_val), (X_test, y_test), class_names, preprocessor
    """
    if not assignments_file.exists():
        raise FileNotFoundError(f"Partition assignments not found at {assignments_file}. Run data_partitioning.py first.")

    print(f"Loading partition assignments from: {assignments_file}")
    df_assign = pd.read_csv(assignments_file)

    print(f"Loading raw features from: {sample_file}")
    # Load original sample using indices from assignment
    df_raw = pd.read_csv(sample_file)

    # Join partition metadata on original index
    df_merged = df_raw.iloc[df_assign["orig_idx"]].copy().reset_index(drop=True)
    df_merged["partition"] = df_assign["partition"].values

    train_df = df_merged[df_merged["partition"] == "train"].copy().reset_index(drop=True)
    val_df = df_merged[df_merged["partition"] == "val"].copy().reset_index(drop=True)

    if load_test:
        test_df = df_merged[df_merged["partition"] == "test"].copy().reset_index(drop=True)
        print(f"Loaded partitions: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}")
    else:
        test_df = None
        print(f"Loaded partitions: Train={len(train_df):,}, Val={len(val_df):,}, Test=[SEALED / UNTOUCHED]")

    # 1. Fit Preprocessor strictly on train
    preprocessor = LeakFreePreprocessor(FEATURE_COLUMNS)
    preprocessor.fit(train_df)

    # 2. Clean features strictly on training and validation
    X_train_raw = preprocessor.clean_transform(train_df)
    X_val_raw = preprocessor.clean_transform(val_df)
    X_test_raw = preprocessor.clean_transform(test_df) if load_test else None

    # 3. Scale features if requested (fitting strictly on train)
    if scale_features:
        X_train = preprocessor.fit_transform_scaler(X_train_raw)
        X_val = preprocessor.transform_scaler(X_val_raw)
        X_test = preprocessor.transform_scaler(X_test_raw) if load_test else None
    else:
        X_train, X_val, X_test = X_train_raw, X_val_raw, X_test_raw

    # 4. Encode labels
    y_train, class_names = encode_target_labels(train_df[TARGET_COLUMN], target_type=target_type)
    y_val, _ = encode_target_labels(val_df[TARGET_COLUMN], target_type=target_type)
    y_test = encode_target_labels(test_df[TARGET_COLUMN], target_type=target_type)[0] if load_test else None

    # 5. Extract validation clean subpopulation mask (D_val_clean, N=72,390)
    from .config import PARTITIONS_DIR
    ablation_assign_file = PARTITIONS_DIR / "ablation_strategy2" / "partition_assignments.csv"
    if ablation_assign_file.exists():
        df_ablation = pd.read_csv(ablation_assign_file)
        clean_val_orig_idx = set(df_ablation[df_ablation["partition"] == "val"]["orig_idx"])
        val_df_orig = df_assign[df_assign["partition"] == "val"]["orig_idx"].values
        val_clean_mask = np.isin(val_df_orig, list(clean_val_orig_idx))
    else:
        val_clean_mask = np.ones(len(val_df), dtype=bool)

    if save_preprocessor_path:
        save_preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessor, save_preprocessor_path)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), class_names, preprocessor, val_clean_mask

