"""
Data loading utilities for CICIoT2023 dataset.
Provides memory-efficient readers, streaming chunk iterators, and dtype optimizations.
"""

from pathlib import Path
from typing import Generator, List, Optional, Union
import numpy as np
import pandas as pd

from src.config import DATASET_DIR, FEATURE_COLUMNS, TARGET_COLUMN


def get_feature_dtypes() -> dict:
    """
    Returns memory-optimized dtypes for all 39 features and label.
    Float32 reduces memory usage by 50% compared to default float64.
    """
    dtypes = {feat: np.float32 for feat in FEATURE_COLUMNS}
    # Protocol Type can be int8 or float32 (CSV sometimes has floats for missing)
    dtypes["Protocol Type"] = np.float32
    dtypes[TARGET_COLUMN] = "category"
    return dtypes


def list_csv_files(dataset_dir: Optional[Union[str, Path]] = None) -> List[Path]:
    """
    Returns sorted list of all CSV files in the dataset directory.
    """
    target_dir = Path(dataset_dir) if dataset_dir else DATASET_DIR
    files = sorted(list(target_dir.glob("*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {target_dir}")
    return files


def stream_csv_chunks(
    file_paths: Optional[List[Path]] = None,
    chunksize: int = 100_000,
    usecols: Optional[List[str]] = None,
) -> Generator[pd.DataFrame, None, None]:
    """
    Generator yielding DataFrame chunks from the given CSV files with memory-efficient dtypes.
    """
    if file_paths is None:
        file_paths = list_csv_files()

    columns_to_load = usecols if usecols is not None else FEATURE_COLUMNS + [TARGET_COLUMN]
    dtypes = {col: get_feature_dtypes()[col] for col in columns_to_load if col in get_feature_dtypes()}

    for fpath in file_paths:
        try:
            for chunk in pd.read_csv(
                fpath,
                usecols=columns_to_load,
                dtype=dtypes,
                chunksize=chunksize,
                low_memory=True,
            ):
                yield chunk
        except Exception as e:
            print(f"Warning: Error reading {fpath.name}: {e}")
            continue


def load_single_csv(
    file_path: Union[str, Path],
    nrows: Optional[int] = None,
    usecols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Loads a single CSV file with memory-efficient dtypes.
    """
    columns_to_load = usecols if usecols is not None else FEATURE_COLUMNS + [TARGET_COLUMN]
    dtypes = {col: get_feature_dtypes()[col] for col in columns_to_load if col in get_feature_dtypes()}
    return pd.read_csv(
        file_path,
        usecols=columns_to_load,
        dtype=dtypes,
        nrows=nrows,
        low_memory=True,
    )
