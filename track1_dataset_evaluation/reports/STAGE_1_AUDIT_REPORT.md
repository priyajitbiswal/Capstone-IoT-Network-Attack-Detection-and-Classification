# Stage 1: Repository Orientation, Structural Audit & Pre-Reconstruction Diagnosis

> Consolidates pre-reconstruction repository orientation (`PROJECT_ORIENTATION_REPORT`), structural/duplicate file audit (`PROJECT_STRUCTURE_AUDIT`), and the Stage 1 forensic audit of the Generation-1 codebase (`track2_ids_engineering/src/`, originally `src/`) vs. the reconstructed leak-free Track 1 pipeline (`track1_dataset_evaluation/src/`, originally `src_v2/`).

---

## 1. Repository Census & Pipeline Generation Comparison

- **Raw Dataset (`data/raw/`)**: 63 CSV files (`Merged01.csv`–`Merged63.csv`), ~8.87 GB (~45M+ bidirectional statistical flows), 40 columns (`39` continuous/discrete features + `Label`).
- **Working Stratified Sample (`data/processed/sample_stratified.csv`)**: **529,562 rows × 40 columns** (106.5 MB), sampled strictly from `Merged01.csv`–`Merged25.csv` (files `26–63` untouched during initial sampling), retaining 100% of 8 minority classes (`UPLOADING_ATTACK`: 416, `RECON-PINGSWEEP`: 722, `BACKDOOR_MALWARE`: 1,061, `XSS`: 1,239, `COMMANDINJECTION`: 1,666, `SQLINJECTION`: 1,718, `BROWSERHIJACKING`: 1,904, `DICTIONARYBRUTEFORCE`: 4,181).
- **Phantom Documentation Fixed**: Legacy docs referenced non-existent `example.ipynb`, `src/utils.py`, and `PROJECT_INSIGHTS.md` (actual file: `INSIGHTS.md`).

| Architectural Dimension | Legacy Generation 1 (`src/`) | Authoritative Leak-Free Track 1 (`track1_dataset_evaluation/src/`) |
|---|---|---|
| **Data Partitioning** | Random 80/20 stratified `train_test_split` | Atomic SHA-256 / 64-bit group-hash partitioning (70.01% / 15.00% / 14.99%) |
| **Feature Leakage** | **10.14%** of test flows match train coordinates (`10,738 / 105,913`) | **0.00%** feature-vector overlap verified across Train, Val, Test & Cross |
| **Validation Split** | None (hyperparameters tuned on test) | Dedicated isolated Validation split (`75,172` rows) |
| **Cross-Capture Holdout** | None (single capture pool) | Independent `Merged26–40` holdout purged of train/val/test vectors (`57,280`) |
| **XGBoost Class Weights** | **Bug**: `class_weight="balanced"` ignored by `XGBClassifier` | Fixed via `BalancedXGBClassifier` (`compute_sample_weight` -> `.fit()`) |
| **Preprocessor Fitting** | `Rate` finite max computed before train/test split | `LeakFreePreprocessor` learns finite maxima + scaler moments strictly on Train |

---

## 2. Schema Audit: CICIoT2023 46 vs. 39 Feature Discrepancy

Original *Neto et al. (Sensors 2023)* lists 46 statistical features (+ label = 47), whereas the raw CSVs in `data/raw/` contain **39 numeric features + `Label` = 40 columns** with IPs, ports, MACs, and timestamps stripped. Seven higher-order/directional metrics (`Magnitude`, `Radius`, `Covariance`, `Weight`, `urg_count`, `Srate`, `Drate`) are absent from the pre-merged CSV distribution processed here.

---

## 3. Code, Benchmarking & Methodological Vulnerabilities Audited

| Severity | Identified Flaw | Root Cause & Empirical Consequence |
|:---:|---|---|
| **CRITICAL** | **10.14% Random-Split Flow Leakage** | `sample_stratified.csv` holds `28,464` exact duplicate rows (`5.38%`) and `59,780` total redundant feature-vector rows (`11.29%`); naive 80/20 splits place `10,738` train-matching flows in test (`81.39%` collision leakage on unique stratum). |
| **CRITICAL** | **Missing Validation Partition** | Generation 1 scripts (`train_*.py`, `experiments/`) evaluated hyperparameters directly against the single 20% test set. |
| **CRITICAL** | **XGBoost Unweighted Training Bug** | `XGBClassifier(class_weight="balanced")` ignores sklearn's `class_weight` unless `sample_weight` is explicitly passed in `.fit()`, making early XGBoost baselines artificially unweighted. |
| **HIGH** | **Batch Throughput vs. Wire-Speed Conflation** | Timing `model.predict(X_scaled)` on pre-scaled 20k RAM batches yields millions of flows/s (`8.71M` LR), hiding sequential Python/tree dispatch overhead (`10–50×` slower per flow), packet parsing, flow-state assembly, and feature extraction. |
| **HIGH** | **Missing Temporal / Cross-Capture Test** | Random splitting scatters correlated packet bursts from the same capture session across train and test. |
| **HIGH** | **Literature Survey Citation Disconnect** | 14 surveyed papers resided only in markdown critiques while the initial LaTeX draft cited only 8 entries (1 dataset paper + 5 toolkits). |
| **MEDIUM** | **Preprocessing Global Bounds Leakage** | `clean_features()` computed global `Rate` finite maxima (`3.35e6`) prior to splitting. |
| **MEDIUM** | **DNN Underfitting & Case Drift** | Legacy PyTorch MLP ran 3–4 epochs; `demo_app.py` mixed-case strings diverged from uppercase CSV labels. |

---

## 4. Prioritized Track 1 Reconstruction Plan Executed in Stages 2–9

1. Deduplicate exact rows (`529,562 -> 501,098`) and build atomic group-hash 70/15/15 Strategy 1 (`350,821 / 75,172 / 75,105`) + collision-purged Strategy 2 ablation (`482,682`) + disjoint `Merged26–40` cross-capture (`57,280`) (**Stage 2**).
2. Fix `BalancedXGBClassifier` + fit-on-train `LeakFreePreprocessor` and train all 21 models using strictly Train & Validation (**Stage 3**).
3. Evaluate all 21 frozen models read-only on Internal Test & Cross-Capture, benchmark P50/P95 single-flow latencies vs. batch throughput, synthesize 14-paper literature review into IEEE/Typst manuscript, and certify all claims (**Stages 4–9**).
