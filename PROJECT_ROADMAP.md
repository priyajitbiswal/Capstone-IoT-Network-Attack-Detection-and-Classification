# Capstone Project Roadmap: IoT Intrusion Detection System (CICIoT2023)

## Project Overview
This project develops an end-to-end Machine Learning and Deep Learning pipeline for IoT Network Intrusion Detection using the **CICIoT2023** benchmark dataset.
Following an incremental, phased approach, models are built and evaluated across three levels of classification granularity:
1. **Phase 1: Binary Classification (2 Classes)** &mdash; `Attack` vs. `Benign`
2. **Phase 2: Functional Category Classification (8 Classes)** &mdash; `DDoS`, `DoS`, `Mirai`, `Recon`, `Spoofing`, `Web`, `BruteForce`, `Benign`
3. **Phase 3: Fine-Grained Attack Classification (34 Classes)** &mdash; All 33 specific attack vectors + `Benign`

> **Note**: As requested, all original files (`example.ipynb`, `MERGED_CSV/`, and existing PDFs) remain strictly untouched. All new modules, notebooks, and scripts will be created in dedicated new directories.

---

## Architecture & Directory Structure

To keep the codebase modular, reusable, and reproducible, new files will follow this structure:

```
capstone/
├── MERGED_CSV/                          # (Original untouched dataset)
├── example.ipynb                        # (Original untouched notebook)
├── PROJECT_ROADMAP.md                   # This roadmap document
├── requirements.txt                     # Project dependencies
├── src/                                 # Modular Python package
│   ├── __init__.py
│   ├── config.py                        # Dataset paths, 39 feature schema, label mappings
│   ├── data_loader.py                   # Chunked/streaming readers, sampling functions
│   ├── preprocessing.py                 # Normalization (StandardScaler), train/test splitters
│   ├── models.py                        # Model definitions (Tree, Boosting, Deep Learning)
│   ├── evaluate.py                      # Metrics, confusion matrix generation, PR curves
│   └── utils.py                         # Plotting helpers, timing and latency profiler
├── data/                                # Generated sample subsets (stratified, balanced)
│   └── sample_stratified.csv            # Manageable stratified working dataset
├── notebooks/                           # Step-by-step interactive notebooks
│   ├── 01_data_exploration_sampling.ipynb
│   ├── 02_phase1_binary_classification.ipynb
│   ├── 03_phase2_eight_class_classification.ipynb
│   └── 04_phase3_thirtyfour_class_classification.ipynb
├── models_saved/                        # Exported model weights/checkpoints (.pkl, .pt)
└── results/                             # Evaluation outputs
    ├── figures/                         # Confusion matrices, ROC/PR curves, feature importance
    └── tables/                          # Metric summaries (Accuracy, Macro F1, Recall, Latency)
```

---

## Detailed Step-by-Step Task Breakdown

```mermaid
graph TD
    Step0[Step 0: Environment & Modular Setup] --> Step1[Step 1: Data Loader & Stratified Sampling]
    Step1 --> Step2[Step 2: Phase 1 - Binary Classification 2-Class]
    Step2 --> Step3[Step 3: Phase 2 - Category Classification 8-Class]
    Step3 --> Step4[Step 4: Phase 3 - Fine-Grained Classification 34-Class]
    Step4 --> Step5[Step 5: Feature Selection & Latency Optimization]
    Step5 --> Step6[Step 6: Comparative Analysis & Final Synthesis]
```

---

### Step 0: Environment & Core Framework Setup
- [x] **Task 0.1**: Create `requirements.txt` specifying exact dependencies:
  - `pandas`, `numpy`, `scipy`
  - `scikit-learn`, `lightgbm`, `xgboost`
  - `torch` (PyTorch)
  - `matplotlib`, `seaborn`
  - `tqdm`, `joblib`, `ipykernel`
- [x] **Task 0.2**: Verify / setup Python virtual environment using `uv venv` and install all dependencies strictly inside `.venv`.
- [x] **Task 0.3**: Initialize project package directories (`src/`, `notebooks/`, `data/`, `models_saved/`, `results/figures/`, `results/tables/`).
- [x] **Task 0.4**: Implement `src/config.py`:
  - Verified and listed the exact **39 features** present in `MERGED_CSV/` (handling `Time_To_Live`, `Rate`, `IGMP`, etc.).
  - Defined label mapping dictionaries for **2 classes** (`Attack` vs. `Benign`).
  - Defined label mapping dictionaries for **8 classes** (`DDoS`, `DoS`, `Mirai`, `Recon`, `Spoofing`, `Web`, `BruteForce`, `Benign`).
  - Defined 34-class label encoder / index mappings (handling all uppercase dataset labels).

---

### Step 1: Data Engineering & Stratified Sampling
- [x] **Task 1.1**: Implement `src/data_loader.py` with memory-safe batch processing:
  - Streaming iterator over the 63 CSV files in `MERGED_CSV/`.
  - Memory-efficient dtype specifications (`float32` for features, categorical label).
- [x] **Task 1.2**: Implement stratified subsampling script (`src/create_sample.py`):
  - Created balanced working sample of **529,562 records** (`data/sample_stratified.csv`, 106.5 MB).
  - Capped majority DDoS/DoS flooding classes while **retaining 100% of all rare classes** (Web attacks, Brute Force, Ping Sweep, Backdoor). All 34 classes verified present.
- [x] **Task 1.3**: Implement `src/preprocessing.py`:
  - Outlier and missing value sanitization.
  - Leak-free `StandardScaler` pipeline fitted strictly on training partition.
  - Stratified 80/20 train/test split with target encoders for 2, 8, and 34 classes.
- [x] **Task 1.4**: Create `notebooks/01_data_exploration_sampling.ipynb` to visualize class distributions and verify pipeline.

---

### Step 2: Phase 1 &mdash; Binary Classification (2 Classes: Attack vs. Benign)
- [x] **Task 2.1**: Define binary classification baseline models in `src/models.py`:
  - Linear Baseline: `SGDClassifier` (log-loss / logistic regression with class weighting).
  - Tree Baseline: `RandomForestClassifier` (optimized for large tabular flows).
  - Gradient Boosted Trees: `LGBMClassifier` & `XGBClassifier`.
  - Neural Network Baseline: `PyTorchTabularDNN` with BatchNorm, Dropout, and AdamW.
- [x] **Task 2.2**: Train and validate binary models:
  - Benchmarked training duration and per-sample inference latency.
  - Recorded performance metrics on 105,913 test instances: Accuracy, Precision, Recall, Macro F1, Weighted F1, ROC-AUC, and PR-AUC.
- [x] **Task 2.3**: Generate binary evaluation reports in `results/`:
  - Generated and saved normalized confusion matrix heatmaps to `results/figures/binary_cm_*.png`.
  - Saved ROC & Precision-Recall curves to `results/figures/binary_roc_pr_*.png`.
  - Saved trained model checkpoints to `models_saved/binary/` (`.joblib` and `.pt`).
  - Generated `results/tables/binary_models_benchmark.csv`.
- [x] **Task 2.4**: Create `notebooks/02_phase1_binary_classification.ipynb` documenting experiments, findings, and decision boundaries. Pre-rendered with all outputs.

---

### Step 3: Phase 2 &mdash; Functional Category Classification (8 Classes)
- [x] **Task 3.1**: Configure 8-class target mapping:
  - Grouping 34 raw labels into: `DDoS`, `DoS`, `Mirai`, `Recon`, `Spoofing`, `Web`, `BruteForce`, and `Benign`.
- [x] **Task 3.2**: Address multi-class category imbalance:
  - Implemented cost-sensitive balanced class weighting (`compute_class_weight`) for all tree and deep neural network models.
- [x] **Task 3.3**: Train and compare candidate models (`src/train_category_8class.py`):
  - Benchmarked SGD Logistic Regression, Random Forest, LightGBM, XGBoost, and PyTorch DNN.
  - Recorded metrics: Accuracy, Macro F1, Weighted F1, Macro Recall, One-vs-Rest ROC-AUC, latency, and throughput.
- [x] **Task 3.4**: Detailed 8-class evaluation:
  - Generated and saved 8&times;8 normalized confusion matrix heatmaps to `results/figures/category8_cm_*.png`.
  - Saved per-class reports for all models to `results/tables/category8_per_class_reports.json`.
- [x] **Task 3.5**: Save best performing 8-class models to `models_saved/category_8class/` (`.joblib` and `.pt`).
- [x] **Task 3.6**: Create and execute `notebooks/03_phase2_eight_class_classification.ipynb` with all rendered tables, plots, and per-class analyses.

---

### Step 4: Phase 3 &mdash; Fine-Grained Attack Classification (34 Classes)
- [x] **Task 4.1**: Configure 34-class label encoder:
  - Full evaluation across all 33 individual attack profiles + Benign traffic.
- [x] **Task 4.2**: Train and compare candidate models (`src/train_attack_34class.py`):
  - Benchmarked SGD Logistic Regression, Random Forest, LightGBM, XGBoost, and PyTorch DNN across all 34 classes.
  - Implemented cost-sensitive balanced class weighting to prevent minority class starvation.
- [x] **Task 4.3**: Comprehensive 34-class evaluation:
  - Full 34-class classification report (accuracy, macro, micro, and weighted averages).
  - High-resolution 34x34 normalized confusion matrix heatmaps saved to `results/figures/attack34_cm_*.png`.
  - Specific deep-dive into stealthy minority vectors:
    - `SQLINJECTION`, `XSS`, `COMMANDINJECTION`, `BROWSERHIJACKING`, `UPLOADING_ATTACK`
    - `DICTIONARYBRUTEFORCE`, `BACKDOOR_MALWARE`, `RECON-PINGSWEEP`
    - Saved minority benchmark table to `results/tables/attack34_minority_attacks_benchmark.csv`.
- [x] **Task 4.4**: Save model artifacts and predictions to `models_saved/attack_34class/`:
  - Exported all 5 model checkpoints (`.joblib` and `.pt`) and `attack34_scaler.joblib`.
  - Exported benchmark table to `results/tables/attack34_models_benchmark.csv` and JSON.
  - Exported per-class reports to `results/tables/attack34_per_class_reports.json`.
- [x] **Task 4.5**: Create and execute `notebooks/04_phase3_thirtyfour_class_classification.ipynb` with all pre-rendered charts, tables, confusion matrix displays, and 2 &rarr; 8 &rarr; 34 progression trajectory.

---

### Step 5: Feature Importance & Inference Latency Optimization
- [x] **Task 5.1**: Feature Importance Analysis (`src/feature_selection.py`):
  - Computed ensemble consensus feature importance across RF Gini, XGBoost Gain, and LightGBM Split.
  - Identified top 20 core features retaining >90% of discriminatory power and identified near-zero gain features (`IRC`, `DHCP`, `SMTP`, `Telnet`, `cwr_flag_number`).
  - Saved rankings to `results/tables/feature_importance_rankings.csv` and chart to `results/figures/feature_importance_ranking.png`.
- [x] **Task 5.2**: Feature Space Reduction:
  - Selected optimal reduced feature subset (**20 core features**).
  - Saved top 20 feature list and reduced scaler to `models_saved/reduced_features/`.
- [x] **Task 5.3**: Trade-off Analysis & Benchmarking:
  - Re-trained candidate tree models (Random Forest, LightGBM, XGBoost) on the reduced 20-feature space.
  - Measured metrics: XGBoost inference latency decreased from 17.01 &mu;s down to 12.15 &mu;s (**+28.55% speedup**, 82,299 flows/sec) with only a &minus;1.96% accuracy delta.
  - Feature vector memory reduced by **48.7%** (156 bytes down to 80 bytes per flow).
  - Saved benchmark table to `results/tables/feature_reduction_benchmark.csv` and trade-off figure to `results/figures/feature_reduction_comparison.png`.
- [x] **Task 5.4**: Create and execute `notebooks/05_feature_selection_latency.ipynb` with all pre-rendered charts, tables, and Pareto trade-off analyses.

---

### Step 6: Capstone Synthesis, Comparison & Presentation
- [ ] **Task 6.1**: Master comparative synthesis:
  - Summary matrix comparing **2-Class vs. 8-Class vs. 34-Class** performance.
  - Comparison table across all model families (Logistic Regression, Random Forest, LightGBM, XGBoost, DNN).
- [ ] **Task 6.2**: Visualizations generation:
  - Multi-panel publication-quality figures saved to `results/figures/`.
  - Performance vs. Complexity / Latency frontier curves.
- [ ] **Task 6.3**: Final Capstone Summary Report:
  - Document key findings, limitations, trade-offs, and defense-ready conclusions.

---

## Progress Tracking Status

| Phase | Granularity | Classes | Status | Target Deliverable |
| :--- | :--- | :---: | :---: | :--- |
| **Step 0: Setup** | Environment & Config | &mdash; |  Completed | `.venv`, `requirements.txt`, `src/config.py` |
| **Step 1: Data** | Sampling & Loader | &mdash; |  Completed | `src/data_loader.py`, `src/create_sample.py`, `data/sample_stratified.csv` |
| **Phase 1** | Binary Detection | 2 |  Completed | `02_phase1_binary_classification.ipynb`, metrics table & models |
| **Phase 2** | Category Detection | 8 |  Completed | `03_phase2_eight_class_classification.ipynb`, 8x8 confusion matrix & models |
| **Phase 3** | Attack Profile Detection | 34 |  Completed | `04_phase3_thirtyfour_class_classification.ipynb`, 34x34 matrix & models |
| **Optimization** | Feature Selection & Latency | 20 feats |  Completed | `05_feature_selection_latency.ipynb`, latency benchmarks & speedup |
| **Synthesis** | Final Capstone Report | All | ⏳ Next Up | Master comparative results & figures |
