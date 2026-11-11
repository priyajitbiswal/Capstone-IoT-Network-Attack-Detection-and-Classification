# Track 2 Data Layout (`track2_ids_engineering/data/`)

Track 2 reads the shared CICIoT2023 raw captures and curated stratified working dataset from [`../../data/`](../../data/):
- `../../data/raw/` (`Merged01.csv`–`Merged63.csv`)
- `../../data/processed/sample_stratified.csv` (529,562 flows across 34 classes)

## Planned Hierarchical IDS Datasets (Design Specification Only)

When implementing the proposed Three-Stage Hierarchical IDS ([`../hierarchical_ids/README.md`](../hierarchical_ids/README.md)), derived training/validation/test datasets will be stored here (gitignored via `*.csv` rules):

- **Dataset A**: Binary labels (`Benign` vs. `Attack`)
- **Dataset B**: Eight attack-category labels
- **Dataset C1**: Flood-category samples only, retaining their original fine-grained attack labels
- **Dataset C2**: All non-flood attack samples, retaining their original fine-grained attack labels

*(Note: When building Datasets A, B, C1, and C2 rigorously, reuse Track 1's leak-free group-hash partition splits from `../../track1_dataset_evaluation/data/partitions/` so train/val/test comparisons against the flat 34-class baseline remain strictly zero-leakage.)*
