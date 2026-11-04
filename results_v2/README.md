# Generation 2 Experimental Results & Metric Ledger (`results_v2/`)

This directory houses the empirical outputs, per-class JSON evaluations, and centralized metrics ledger for the leak-free Generation 2 research pipeline.

> [!NOTE]
> All Stage 1 through Stage 7 markdown reports have been consolidated into the dedicated [`reports/`](../reports/) directory for unified research provenance.

---

## Directory Contents

### 1. `tables/`
- **[`centralized_experiment_ledger.csv`](tables/centralized_experiment_ledger.csv)**: Master benchmark ledger logging validation Accuracy, Macro-F1, Weighted-F1, single-packet inference latency, throughput (PPS), peak memory consumption, and training duration across all 21 models.
- **[`stage4_test_results.csv`](tables/stage4_test_results.csv)**: Consolidated Stage 4 test evaluation table reporting Internal Test ($N=75,105$) and same-testbed Temporal Cross-Capture Test ($N=57,280$) metrics, including Macro-F1, Weighted-F1, Accuracy, MCC, ROC-AUC, and Functional Threat-Tier F1 scores.

### 2. `metrics/`
- **`per_class/`**: 21 individual JSON files containing granular per-class Precision, Recall, F1-Score, and Support for every model.
- **`stage4/`**: 21 master evaluation JSON files containing overall and per-tier metrics for both internal and cross-capture test sets.
- **Loss Logs**: `binary_pytorch_dnn_dnn_loss.json`, `category_pytorch_dnn_dnn_loss.json`, `fine_grained_pytorch_dnn_dnn_loss.json` recording epoch-by-epoch training and validation loss curves.

### 3. `figures/`
- Reserved output directory for Generation 2 generated ROC, PR, and confusion matrix visualizations.
