# Data Architecture & Partitioning Schema (`data/`)

This directory houses the dataset partitions, processed samples, and raw capture repositories for the CICIoT2023 Intrusion Detection System.

---

## Directory Organization

```text
data/
├── raw/                              # 63 full-scale CICIoT2023 captures (Merged01.csv - Merged63.csv, 8.87 GB)
├── processed/                        # Curated working samples
│   └── sample_stratified.csv         # 529,562 flows (capped floods at 20k, 100% rare attack retention)
├── partitions/                       # Stage 2 leak-free feature-hash grouped partitions (Strategy 1 Primary)
│   ├── partition_assignments.csv     # Mapping each row in sample_stratified.csv to Train/Val/Test
│   ├── partition_metadata.json       # Distribution statistics, collision counts, split sizes
│   ├── cross_capture_test.csv        # Temporal test partition from Merged26-Merged40 (63,556 flows)
│   ├── cross_capture_test_clean.csv  # Purged temporal test partition with 0% train collision overlap (57,280 flows)
│   └── cross_capture_metadata.json   # Session breakdown and purge statistics
└── partitions/ablation_strategy2/    # Strategy 2 collision-purged partitions
    ├── partition_assignments.csv     # Purged assignments removing all 8,740 conflicting hashes
    └── metadata.json                 # Stratification verification and class counts
```

---

## Partitioning Integrity & Zero Leakage

To eliminate identical-feature data leakage, flows are hashed using SHA-256 over their 39 numerical features:
1. **Strategy 1 (Primary Benchmark)**: Feature hashes are grouped atomically. All flows sharing an identical feature vector reside in exactly one partition (Train: 350,821; Val: 75,172; Test: 75,105). Zero feature vector overlap exists between splits.
2. **Strategy 2 (Ambiguity-Purged Ablation)**: All 8,740 unique feature hashes associated with conflicting labels across classes ($N=54,828$ rows) are completely purged from the dataset (Train: 337,838; Val: 72,390; Test: 72,454).
3. **Cross-Capture Clean Test**: Sampled from subsequent recording sessions (`Merged26.csv` through `Merged40.csv`), and filtered to remove any feature hashes present in the training set ($N=57,280$).

> [!NOTE]
> All raw CSVs, processed samples, and partition assignment tables are gitignored to preserve bandwidth and repository hygiene. They are generated deterministically via [`src_v2/data_partitioning.py`](../src_v2/data_partitioning.py) and [`src_v2/cross_capture.py`](../src_v2/cross_capture.py).
