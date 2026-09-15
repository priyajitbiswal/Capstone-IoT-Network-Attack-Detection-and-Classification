"""
Stratified sampling script for CICIoT2023.
Extracts a balanced, representative working dataset from MERGED_CSV files.
Retains 100% of rare attack vectors while undersampling massive flooding attacks.
"""

import argparse
from collections import Counter
from pathlib import Path
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import DATA_DIR, DATASET_DIR, SAMPLE_FILE, TARGET_COLUMN
from src.data_loader import get_feature_dtypes, list_csv_files


def create_stratified_sample(
    max_per_class: int = 25_000,
    benign_cap: int = 50_000,
    max_files: int = 25,
    output_path: Path = SAMPLE_FILE,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Builds a stratified sample by streaming CSV files.
    
    Args:
        max_per_class: Maximum number of samples to retain for each attack class.
        benign_cap: Maximum number of benign samples to retain (kept higher to ensure benign visibility).
        max_files: Maximum number of MERGED_CSV files to process.
        output_path: Destination path for the saved CSV sample.
        random_state: Random seed for shuffling.
        
    Returns:
        pd.DataFrame: The aggregated stratified dataset.
    """
    csv_files = list_csv_files(DATASET_DIR)[:max_files]
    print(f"Starting stratified sampling across {len(csv_files)} files...")
    print(f"Target caps: Max {max_per_class:,} per attack class | Max {benign_cap:,} for BENIGN")
    print(f"Rare attack classes will be retained at 100% representation.")

    class_buffers = {}
    class_counts = Counter()

    dtypes = get_feature_dtypes()
    start_time = time.time()

    for file_idx, fpath in enumerate(tqdm(csv_files, desc="Processing CSV files")):
        # Read file with optimized dtypes
        df_chunk = pd.read_csv(fpath, dtype=dtypes, low_memory=True)
        
        # Group by Label
        for label, group in df_chunk.groupby(TARGET_COLUMN, observed=False):
            cap = benign_cap if str(label).upper() == "BENIGN" else max_per_class
            current_count = class_counts[label]
            needed = cap - current_count
            
            if needed <= 0:
                continue
            
            # If group has more than needed, take a random sample
            if len(group) > needed:
                selected = group.sample(n=needed, random_state=random_state + file_idx)
            else:
                selected = group
                
            if label not in class_buffers:
                class_buffers[label] = []
            class_buffers[label].append(selected)
            class_counts[label] += len(selected)

        del df_chunk

    # Concatenate all collected samples
    print("\nConcatenating sampled classes...")
    all_dfs = []
    for label, dfs in class_buffers.items():
        all_dfs.extend(dfs)

    sample_df = pd.concat(all_dfs, ignore_index=True)
    
    # Shuffle the final dataset
    sample_df = sample_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    elapsed = time.time() - start_time
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nWriting sample to {output_path}...")
    sample_df.to_csv(output_path, index=False)
    file_size_mb = output_path.stat().st_size / (1024 * 1024)

    print(f"\n========================================================")
    print(f" Stratified Sampling Complete in {elapsed:.1f} seconds")
    print(f" Total Rows Extracted : {len(sample_df):,}")
    print(f" Total Classes Found   : {sample_df[TARGET_COLUMN].nunique()} of 34")
    print(f" Output File Size      : {file_size_mb:.1f} MB ({output_path.name})")
    print(f"========================================================")

    print("\nSampled Class Distribution:")
    counts = sample_df[TARGET_COLUMN].value_counts()
    for lbl, count in counts.items():
        pct = (count / len(sample_df)) * 100
        print(f"  {lbl:<28}: {count:>8,} ({pct:5.2f}%)")

    return sample_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create stratified sample from CICIoT2023 CSV files.")
    parser.add_argument("--max-per-class", type=int, default=25_000, help="Max instances per attack class")
    parser.add_argument("--benign-cap", type=int, default=50_000, help="Max instances for BENIGN class")
    parser.add_argument("--max-files", type=int, default=25, help="Number of MERGED CSV files to sample from")
    parser.add_argument("--output", type=str, default=str(SAMPLE_FILE), help="Output path for sample CSV")
    args = parser.parse_args()

    create_stratified_sample(
        max_per_class=args.max_per_class,
        benign_cap=args.benign_cap,
        max_files=args.max_files,
        output_path=Path(args.output),
    )
