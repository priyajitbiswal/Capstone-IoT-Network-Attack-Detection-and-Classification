"""
Module for extracting an independent Cross-Capture Holdout dataset from Merged26.csv-Merged63.csv.
Guarantees absolute zero overlap with the primary training partitions.
"""

from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

from src_v2.config import (
    DATASET_DIR,
    FEATURE_COLUMNS,
    PARTITIONS_DIR,
    RANDOM_SEED,
    TARGET_COLUMN,
)


CROSS_CAPTURE_FILE = PARTITIONS_DIR / "cross_capture_test.csv"
CROSS_CAPTURE_METADATA_FILE = PARTITIONS_DIR / "cross_capture_metadata.json"


def extract_cross_capture_sample(
    max_per_class: int = 2_000,
    benign_cap: int = 5_000,
    files_to_scan: int = 15,
    random_seed: int = RANDOM_SEED,
    output_path: Path = CROSS_CAPTURE_FILE,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Extracts an independent cross-capture evaluation sample from Merged26.csv onwards.
    """
    all_csvs = sorted(list(DATASET_DIR.glob("*.csv")))
    # Files Merged26 to Merged63 (indices 25 to 62)
    cross_capture_files = all_csvs[25: 25 + files_to_scan]

    print(f"Sampling cross-capture test flows across {len(cross_capture_files)} unseen CSV files:")
    print(f"  From {cross_capture_files[0].name} to {cross_capture_files[-1].name}")

    class_buffers: Dict[str, List[pd.DataFrame]] = {}
    class_counts: Counter = Counter()

    dtypes = {c: np.float32 for c in FEATURE_COLUMNS}
    dtypes[TARGET_COLUMN] = "category"

    cols_to_load = FEATURE_COLUMNS + [TARGET_COLUMN]

    for fpath in tqdm(cross_capture_files, desc="Scanning cross-capture files"):
        df_chunk = pd.read_csv(fpath, usecols=cols_to_load, dtype=dtypes, low_memory=True)

        for label, group in df_chunk.groupby(TARGET_COLUMN, observed=False):
            lbl_str = str(label).strip().upper()
            cap = benign_cap if lbl_str == "BENIGN" else max_per_class
            curr = class_counts[lbl_str]
            needed = cap - curr

            if needed <= 0:
                continue

            if len(group) > needed:
                selected = group.sample(n=needed, random_state=random_seed)
            else:
                selected = group

            if lbl_str not in class_buffers:
                class_buffers[lbl_str] = []
            class_buffers[lbl_str].append(selected)
            class_counts[lbl_str] += len(selected)

        del df_chunk

    # Concatenate sampled classes
    all_dfs = []
    for lbl, dfs in class_buffers.items():
        all_dfs.extend(dfs)

    df_cross = pd.concat(all_dfs, ignore_index=True)
    df_cross = df_cross.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    # Save to partitions dir
    PARTITIONS_DIR.mkdir(parents=True, exist_ok=True)
    df_cross.to_csv(output_path, index=False)

    metadata = {
        "source_files": [f.name for f in cross_capture_files],
        "total_records": len(df_cross),
        "classes_found": int(df_cross[TARGET_COLUMN].nunique()),
        "class_distribution": {k: int(v) for k, v in df_cross[TARGET_COLUMN].value_counts().items()},
        "random_seed": random_seed,
    }

    import json
    with open(CROSS_CAPTURE_METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nExtracted {len(df_cross):,} cross-capture evaluation flows across {df_cross[TARGET_COLUMN].nunique()} classes.")
    print(f"Saved to: {output_path}")

    return df_cross, metadata


if __name__ == "__main__":
    extract_cross_capture_sample()
