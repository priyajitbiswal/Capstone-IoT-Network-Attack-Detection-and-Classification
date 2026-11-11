# Track 1 Dataset-Distribution Experiments (`track1_dataset_evaluation/experiments/`)

These two standalone scripts examine how CICIoT2023's raw heavy-tailed traffic distribution (>85%–90% volumetric DDoS/DoS floods) produces ~99% headline accuracy in literature compared with stratified balanced evaluation:

- **`natural_distribution_evaluation.py`**: Reconciles model behavior under stratified balanced sampling vs. natural CICIoT2023 class priors.
- **`uncapped_pure34_benchmark.py`**: Evaluates pure 34-class classification on raw uncapped `MERGED_CSV` streams without majority flood capping.

> **Paths**: Both scripts read the shared dataset at [`../../data/processed/sample_stratified.csv`](../../data/processed/sample_stratified.csv) (and [`../../data/raw/`](../../data/raw/)) and write their outputs (`exp5_distribution_reconciliation.csv`, `exp7_uncapped_pure34.csv`) into [`../results/tables/`](../results/tables/).
