# Track 1 Experimental Results & Metric Ledger (`track1_dataset_evaluation/results/`)

This directory houses the empirical outputs, per-class JSON evaluations, distribution-audit CSVs, and centralized metrics ledger for Track 1 (CICIoT2023 Dataset Integrity & Benchmark Evaluation).

> [!NOTE]
> All Stage 1 through Stage 9 markdown reports, audits, and blueprints are stored in [`../reports/`](../reports/) for unified research provenance.

---

## Directory Contents

### 1. `tables/`
- **[`centralized_experiment_ledger.csv`](tables/centralized_experiment_ledger.csv)**: Master benchmark ledger logging validation Accuracy, Macro-F1, Weighted-F1, single-packet inference latency, throughput (PPS), peak memory consumption (RSS), and training duration across all 21 models.
- **[`stage4_test_results.csv`](tables/stage4_test_results.csv)**: Consolidated Stage 4 test evaluation table reporting Internal Test (N = 75,105) and same-testbed Temporal Cross-Capture Test (N = 57,280) metrics, including Macro-F1, Weighted-F1, Accuracy, MCC, ROC-AUC, and Functional Threat-Tier F1 scores.
- **[`exp5_distribution_reconciliation.csv`](tables/exp5_distribution_reconciliation.csv)** & **[`exp7_uncapped_pure34.csv`](tables/exp7_uncapped_pure34.csv)**: Outputs from the Track 1 natural-distribution and uncapped 34-class dataset experiments ([`../experiments/`](../experiments/)).

### 2. `metrics/`
- **`per_class/`**: 21 individual JSON files containing granular per-class Precision, Recall, F1-Score, and Support for every model.
- **`stage4/`**: 21 master evaluation JSON files containing overall and per-tier metrics for both internal and cross-capture test sets.
- **Loss Logs**: `binary_pytorch_dnn_dnn_loss.json`, `category_pytorch_dnn_dnn_loss.json`, `fine_grained_pytorch_dnn_dnn_loss.json` recording epoch-by-epoch training and validation loss curves.

### 3. `figures/`
- Destination directory for Track 1 dataset evaluation visualizations. Publication figures embedded in the manuscript reside in [`../paper/figures/`](../paper/figures/).
