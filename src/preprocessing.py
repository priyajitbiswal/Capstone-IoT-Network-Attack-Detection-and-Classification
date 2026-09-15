"""
Data preprocessing, cleaning, scaling, and stratified splitting pipeline.
Ensures zero data leakage by fitting transformers strictly on the training partition.
"""

from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (
    BINARY_CLASSES,
    BINARY_TO_IDX,
    EIGHT_CLASSES,
    EIGHT_TO_IDX,
    FEATURE_COLUMNS,
    IDX_TO_BINARY,
    IDX_TO_EIGHT,
    IDX_TO_THIRTY_FOUR,
    LABEL_MAPPING_8CLASSES,
    LABEL_MAPPING_BINARY,
    MODELS_DIR,
    TARGET_COLUMN,
    THIRTY_FOUR_CLASSES,
    THIRTY_FOUR_TO_IDX,
)


def clean_features(df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Sanitizes feature columns by replacing infinite values and handling missing entries.
    """
    cols = feature_cols if feature_cols is not None else FEATURE_COLUMNS
    X = df[cols].copy()

    # Replace inf and -inf with NaN
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Fill NaN with 0.0 or column median
    if X.isna().any().any():
        X.fillna(0.0, inplace=True)

    return X.astype(np.float32)


def encode_labels(
    df: pd.DataFrame,
    target_type: Literal["binary", "category", "fine_grained"] = "binary",
) -> Tuple[np.ndarray, Dict, List[str]]:
    """
    Encodes raw dataset labels according to target granularity:
      - 'binary': 2 classes (0: Benign, 1: Attack)
      - 'category': 8 classes (0..7 functional categories)
      - 'fine_grained': 34 classes (0..33 individual attack profiles + Benign)
      
    Returns:
        y (np.ndarray): Integer-encoded target labels.
        mapping_dict (Dict): Dictionary used for conversion.
        class_names (List[str]): Ordered list of class names.
    """
    raw_labels = df[TARGET_COLUMN].astype(str).str.strip().str.upper()

    if target_type == "binary":
        # Map raw label to Attack / Benign string then to int
        mapped_labels = raw_labels.map(LABEL_MAPPING_BINARY)
        # Fallback for any unexpected label
        mapped_labels = mapped_labels.fillna("Attack")
        y = mapped_labels.map(BINARY_TO_IDX).values.astype(np.int64)
        return y, BINARY_TO_IDX, BINARY_CLASSES

    elif target_type == "category":
        # Map raw label to 8 functional groups then to int
        mapped_labels = raw_labels.map(LABEL_MAPPING_8CLASSES)
        mapped_labels = mapped_labels.fillna("DDoS")
        y = mapped_labels.map(EIGHT_TO_IDX).values.astype(np.int64)
        return y, EIGHT_TO_IDX, EIGHT_CLASSES

    elif target_type == "fine_grained":
        # Map raw label directly to 34 class index
        y = raw_labels.map(THIRTY_FOUR_TO_IDX).values.astype(np.int64)
        return y, THIRTY_FOUR_TO_IDX, THIRTY_FOUR_CLASSES

    else:
        raise ValueError(f"Unknown target_type: {target_type}. Must be 'binary', 'category', or 'fine_grained'.")


def prepare_dataset(
    df: pd.DataFrame,
    target_type: Literal["binary", "category", "fine_grained"] = "binary",
    test_size: float = 0.2,
    random_state: int = 42,
    scale_features: bool = True,
    save_scaler_path: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], Optional[StandardScaler]]:
    """
    Full preprocessing pipeline:
      1. Cleans feature columns
      2. Encodes target labels according to granularity
      3. Performs stratified train/test split
      4. Fits StandardScaler strictly on training set (preventing data leakage)
      5. Optionally saves fitted scaler for inference
      
    Returns:
        X_train, X_test, y_train, y_test, class_names, scaler
    """
    X = clean_features(df).values
    y, _, class_names = encode_labels(df, target_type=target_type)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scaler = None
    if scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_test = scaler.transform(X_test).astype(np.float32)

        if save_scaler_path:
            save_scaler_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(scaler, save_scaler_path)

    return X_train, X_test, y_train, y_test, class_names, scaler
