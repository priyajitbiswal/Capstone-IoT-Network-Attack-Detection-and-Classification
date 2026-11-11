# CICIoT2023 End-to-End Reproducibility Guide

This document provides the complete, authoritative guide for independently cloning this repository, obtaining or reconstructing the dataset, regenerating leak-free partitions, retraining models, reproducing all experimental results, and rebuilding the research paper.

---

## 1. System Requirements & Environment Setup

### Hardware Specifications
- **Operating System**: Linux (Ubuntu 22.04+ recommended), macOS, or Windows 10/11 (64-bit).
- **Processor**: Modern multi-core x86_64 CPU (e.g., Intel Core i5/i7/i9 or AMD Ryzen).
- **System Memory (RAM)**: Minimum 16 GB RAM required for dataset deduplication and feature hashing; 32 GB recommended.
- **Storage**:
  - ~150 MB for the base cloned repository (including paper, reports, scripts, and results).
  - ~15 GB free disk space if downloading and processing all raw CICIoT2023 CSV files.
  - ~500 MB if starting directly from `sample_stratified.csv`.

### Python Environment Installation
The codebase requires **Python 3.10+** (tested and verified on Python 3.13).

```bash
# 1. Clone the repository
git clone https://github.com/<username>/ciciot2023-intrusion-detection.git
cd ciciot2023-intrusion-detection

# 2. Create and activate a clean virtual environment
python -m venv .venv

# On Linux / macOS:
source .venv/bin/activate

# On Windows (PowerShell):
.\.venv\Scripts\activate

# 3. Upgrade pip and install all required dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2. Dataset Acquisition & Organization

The research evaluates the **CICIoT2023** benchmark published by the Canadian Institute for Cybersecurity (UNB):
- **Official Dataset Portal**: [https://www.unb.ca/cic/datasets/iot-dataset-2023.html](https://www.unb.ca/cic/datasets/iot-dataset-2023.html)
- **License**: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0).

### Expected Directory Hierarchy
Download the zipped CSV dataset from the official portal and extract the CSV files into `data/raw/` (or create a symlink / directory named `MERGED_CSV/`):

```
data/
├── raw/
│   ├── Merged01.csv   # Required for Stratified Sampling (Files 01–25)
│   ├── Merged02.csv
│   │   ...
│   ├── Merged25.csv
│   ├── Merged26.csv   # Required for Cross-Capture Holdout (Files 26–40)
│   │   ...
│   └── Merged40.csv
└── README.md
```

### Deterministic Stratified Sample Regeneration
Because the full 46-million-record benchmark is prohibitively large for repeated training (~9.1 GB across 63 files), our methodology employs a representative, class-balanced stratified sample preserving 100% of minority attacks.

To regenerate `data/processed/sample_stratified.csv` from raw CSV files:
```bash
python -m track2_ids_engineering.src.create_sample
```
- **Execution Time**: ~2–4 minutes depending on disk I/O.
- **Expected Output**: `data/processed/sample_stratified.csv`
- **Verified Record Count**: Exactly **529,562 rows** × **40 columns** (39 numerical features + 1 ground-truth `Label`).
- **Cryptographic Verification**:
  ```bash
  # Verify SHA-256 checksum of generated sample
  python -c "import hashlib; print(hashlib.sha256(open('data/processed/sample_stratified.csv','rb').read()).hexdigest())"
  ```
  Expected Checksum: `39370152e9b2cf95e64e1df650d3bc677241f811cd38983894502fd1f8baa227`

*(Note: In accordance with GitHub repository size guidelines, the 106.43 MB CSV file is excluded from Git tracking. If an external institutional repository mirror is configured, download the verified CSV directly to `data/processed/sample_stratified.csv` and verify the checksum above).*

---

## 3. End-to-End Execution Sequence (Track 1)

Execute the pipeline from repository root in the strict sequential order documented below:

### Step 1: Zero-Leakage Atomic Hash-Grouped Partitioning (Strategy 1)
Partitions the deduplicated unique stratum (N = 501,098) into strictly disjoint Train (70%), Validation (15%), and Internal Test (15%) splits using SHA-256 feature-coordinate hashing with random seed 42:
```bash
python -m track1_dataset_evaluation.src.data_partitioning
```
- **Input**: `data/processed/sample_stratified.csv`
- **Generated Outputs**:
  - `track1_dataset_evaluation/data/partitions/partition_assignments.csv` (501,098 partition mapping rows)
  - `track1_dataset_evaluation/data/partitions/partition_metadata.json` (class distributions and split counts)
- **Execution Time**: ~30 seconds.

### Step 2: Collision-Purged Partitioning (Strategy 2 Ablation)
Purges all 8,740 conflicting-label collision coordinate hashes (18,416 rows) to construct the separable ablation stratum (N = 482,682):
```bash
python -m track1_dataset_evaluation.src.ablation_strategy2
```
- **Input**: `track1_dataset_evaluation/data/partitions/partition_assignments.csv`
- **Generated Outputs**:
  - `track1_dataset_evaluation/data/partitions/ablation_strategy2/partition_assignments.csv` (482,682 rows)
  - `track1_dataset_evaluation/data/partitions/ablation_strategy2/metadata.json`
- **Execution Time**: ~20 seconds.

### Step 3: Independent Cross-Capture Holdout Extraction & Cleaning
Samples unseen sequential capture files (`Merged26.csv` through `Merged40.csv`) and strictly purges any feature coordinates appearing in training, validation, or test sets:
```bash
python -m track1_dataset_evaluation.src.cross_capture
```
- **Inputs**: `data/raw/Merged26.csv`–`Merged40.csv`, Strategy 1 partition assignments
- **Generated Outputs**:
  - `track1_dataset_evaluation/data/partitions/cross_capture_test.csv` (63,423 raw flows)
  - `track1_dataset_evaluation/data/partitions/cross_capture_test_clean.csv` (Exactly **57,280 clean, uncontaminated flows**)
  - `track1_dataset_evaluation/data/partitions/cross_capture_metadata.json`
- **Execution Time**: ~1–2 minutes.

### Step 4: Strict Multi-Partition Independence Verification
Validates that feature leakage is strictly 0.00% across all 6 pairwise partition intersections:
```bash
python -m track1_dataset_evaluation.src.verify_partitions
```
- **Checks Executed**:
  1. 64-bit MurmurHash3 feature hash disjointness across all 6 partition pairs.
  2. Exact 39-dimensional floating-point coordinate tuple equality across all 6 pairs.
- **Expected Terminal Output**:
  ```
  VERIFICATION COMPLETE: Zero duplicate feature-vector overlap verified.
  All four partitions are strictly mutually disjoint.
  ```
- **Exit Code**: `0` on success (raises `AssertionError` if any overlap exists).

### Step 5: Model Training (21 Baseline & Ablation Configurations)
Fits preprocessors strictly on the training partition and trains all 5 model families across Binary, Category, and Fine-Grained granularities:
```bash
python -m track1_dataset_evaluation.src.run_stage3
```
- **Architectures**:
  - SGD Logistic Regression (`loss='log_loss'`, L2 regularization)
  - Random Forest (`n_estimators=50`, `max_depth=20`)
  - LightGBM (`boosting_type='gbdt'`, `num_leaves=63`, `n_estimators=100`)
  - XGBoost (`hist` tree method, balanced weighting)
  - PyTorch Deep MLP (3-layer feedforward architecture with batch normalization and dropout)
- **Experimental Conditions**:
  - Strategy 1 (Cost-sensitive class balanced weights, 3 granularities × 5 models = 15 models)
  - Strategy 2 Ablation (Collision-purged clean training, 4 models)
  - Unweighted Loss Ablation (Empirical risk minimization, 2 models)
- **Output Checkpoints**: Saved to `track1_dataset_evaluation/models/` alongside fitted `RobustScaler` preprocessors.
- **Execution Time**: ~12–18 minutes on an 8-core CPU.

### Step 6: Multi-Granularity Model Evaluation & Holdout Assessment
Evaluates all trained models across Validation, Internal Test, and Clean Cross-Capture Holdout:
```bash
python -m track1_dataset_evaluation.src.run_stage4_eval
```
- **Generated Outputs**:
  - Detailed per-class precision, recall, F1, and support JSONs in `track1_dataset_evaluation/results/metrics/per_class/`
  - Granular stage 4 test metrics in `track1_dataset_evaluation/results/metrics/stage4/`
  - Consolidated CSV tables:
    - `track1_dataset_evaluation/results/tables/stage4_test_results.csv`
    - `track1_dataset_evaluation/results/tables/centralized_experiment_ledger.csv`
- **Execution Time**: ~2–3 minutes.

### Step 7: Host CPU Latency & Throughput Hardware Profiling (Track 2)
Benchmarks per-flow P50 inference latency and batch throughput across Track 2 baseline model architectures:
```bash
python track2_ids_engineering/scripts/benchmark_throughput.py
```
- **Execution Time**: ~2 minutes.

### Step 8: Publication Figures Regeneration (Track 1)
Generates publication-quality figures:
```bash
python track1_dataset_evaluation/scripts/generate_paper_figures.py
```
- **Generated Outputs**:
  - `track1_dataset_evaluation/paper/figures/fig1_partitioning_protocol.png` (Group-hash protocol schematic)
  - `track1_dataset_evaluation/paper/figures/fig2_accuracy_vs_macro_f1.png` (Diagnostic boundary scatter plot)
  - `track1_dataset_evaluation/paper/figures/fig3_minority_recall_collapse.png` (Weighting trade-off bar chart)
  - `track1_dataset_evaluation/paper/figures/fig4_threat_tier_cross_capture.png` (Threat-tier cross-capture chart)
  - `track1_dataset_evaluation/paper/figures/fig5_latency_pareto_frontier.png` (Latency vs. Macro-F1 Pareto frontier)
- **Execution Time**: < 15 seconds.

### Step 9: Research Paper Compilation (Track 1)
Compiles the complete manuscript deliverable using the native Typst compiler:
```bash
python -c "import typst; typst.compile('track1_dataset_evaluation/paper/paper.typ', output='track1_dataset_evaluation/paper/research_paper.pdf')"
```
- **Primary Source**: `track1_dataset_evaluation/paper/paper.typ`
- **Mirrored LaTeX Source**: `track1_dataset_evaluation/paper/main.tex` (IEEEtran format)
- **Bibliography**: `track1_dataset_evaluation/paper/references.bib`
- **Compiled Output**: `track1_dataset_evaluation/paper/research_paper.pdf`
- **Execution Time**: < 3 seconds.

---

## 4. Expected Results Verification Matrix

To verify that local execution reproduced the exact published benchmark findings, compare `track1_dataset_evaluation/results/tables/stage4_test_results.csv` against these verified reference figures:

| Granularity | Top Architecture | Test Accuracy | Macro-F1 | Weighted-F1 | Cross-Capture Macro-F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Binary (2-class)** | Random Forest | 89.61% | 0.7690 | 0.9038 | 0.7029 |
| **Binary (2-class)** | LightGBM | 89.15% | 0.7637 | 0.9004 | 0.7034 |
| **Category (8-class)** | LightGBM | 78.93% | 0.6894 | 0.7967 | 0.5488 |
| **Category (8-class)** | Random Forest | 78.73% | 0.6781 | 0.7946 | 0.5471 |
| **Fine-Grained (34-class)** | LightGBM | 72.65% | 0.6145 | 0.7327 | 0.4421 |
| **Fine-Grained (34-class)** | XGBoost | 72.48% | 0.6120 | 0.7314 | 0.4410 |

### Minority Attack Recall Under Class Weighting (Fine-Grained)
- **SQL Injection Recall**: 35.83% (Cost-Sensitive Weighted) vs. 7.48% (Unweighted Loss) → **79.12% relative recall collapse**.
- **XSS Attack Recall**: 20.22% (Cost-Sensitive Weighted) vs. 3.83% (Unweighted Loss) → **81.06% relative recall collapse**.
- **Headline Accuracy**: 72.65% (Weighted) vs. 74.45% (Unweighted) → **+1.80% headline accuracy increase at the expense of minority detection**.

---

## 5. Artifact Storage & Git Inclusion Policy

To maintain a lean repository (< 15 MB excluding research papers) and ensure 100% compatibility with GitHub limits:

- **Tracked in Git**:
  - All source code (`track1_dataset_evaluation/src/`, `track2_ids_engineering/src/`, `experiments/`, `scripts/`)
  - All documentation (`README.md`, `PROJECT_STRUCTURE.md`, `REPRODUCIBILITY.md`, `TRACKS_OVERVIEW.md`)
  - All historical research reports (`track1_dataset_evaluation/reports/STAGE_1_...` through `STAGE_9_...`)
  - All evaluation results and metrics (`track1_dataset_evaluation/results/metrics/`, `track1_dataset_evaluation/results/tables/`, `track2_ids_engineering/results/`)
  - All manuscript sources, bibliography, figures, and compiled PDF (`track1_dataset_evaluation/paper/`)
  - Partition metadata JSONs (`track1_dataset_evaluation/data/partitions/*.json`)
- **Excluded from Git (`.gitignore`)**:
  - Raw multi-gigabyte CSVs (`data/raw/*.csv`)
  - Working sample CSVs (`data/processed/sample_stratified.csv`, `data/*.csv`)
  - Generated partition CSVs (`track1_dataset_evaluation/data/partitions/*.csv`)
  - Compiled model binaries and weights (`track1_dataset_evaluation/models/*.joblib`, `track2_ids_engineering/models/**/*`, `*.pt`)
  - Python virtual environments (`.venv/`) and local bytecode caches (`__pycache__/`)
