# Project Structure Audit & Repository Architecture Report

**Repository Root**: `C:\Users\priya\Downloads\capstone`  
**Audit Type**: Complete Read-Only Systemic Audit  
**Total Tracked/Project Files (excl. `.git`, `.venv`)**: 357 files across 34 directories  
**Total Project Volume on Disk (excl. `.git`, `.venv`)**: 10,121,822,067 bytes (9.43 GB / 9652.92 MB)  
**Active Git Branch**: `main` | **Working-Tree Status**: `Clean`  
**HEAD Commit**: `2d5a6c5 docs(stage7): calibrate literature claims and address overclaims`  

---

## 1. Executive Overview & Repository Health

This repository hosts an end-to-end Machine Learning, Deep Learning, and Methodological Auditing investigation for IoT Network Intrusion Detection evaluated on the **CICIoT2023** benchmark (Canadian Institute for Cybersecurity, 105 IoT devices, 33 attack types).

The codebase represents a mature, two-generation research progression:
1. **Generation 1 (`src/`, `models_saved/`, `results/`, `notebooks/`)**: Original exploratory pipeline implementing random 80/20 train/test splits, single-model and multi-model baselines (SGD, RF, LightGBM, XGBoost, PyTorch MLP), feature selection (reducing 39 to 20 features), wire-speed throughput benchmarking, and Streamlit demo.
2. **Generation 2 (`src_v2/`, `models_saved_v2/`, `results_v2/`, `data/partitions/`)**: Rigorous, leak-free re-architecture executing Stages 1 through 7:
   - **Stage 1 (Data Audit)**: Audited 529,562 flows, quantifying exact duplicate records, identical-feature vectors, and 8,740 conflicting label collision hashes.
   - **Stage 2 (Partitioning)**: Implemented atomic feature-hash grouped partitioning enforcing zero vector overlap across splits (Strategy 1 Primary, Strategy 2 Purged, and Cross-Capture Clean).
   - **Stage 3 (Model Suite)**: Trained and audited 21 baseline model checkpoints across 5 phases with stateful `LeakFreePreprocessor` serialization.
   - **Stage 4 (Evaluation)**: Evaluated all 21 models strictly read-only across Internal Test (`N = 75,105`) and same-testbed Temporal Cross-Capture Test (`N = 57,280`).
   - **Stage 5 (Error Analysis)**: Comprehensive confusion, false-negative, and ablation error analysis.
   - **Stage 6 (Research Synthesis)**: Rigorous synthesis resolving scientific claims, trade-offs, and calibrated findings.
   - **Stage 7 (Literature Review)**: Deep-dive review of 14 peer-reviewed research papers and capstone technical documentation.

---

## 2. Git Status, Branching, and Commit State

- **Current Branch**: `main`
- **Working Tree**: `Clean (no unstaged or untracked project changes)`
- **Branch Upstream State**: Ahead of `origin/main` by 6 local commits.
- **Recent Git Commits**:
  ```text
  2d5a6c5 docs(stage7): calibrate literature claims and address overclaims
  7e853fd docs(stage7): add literature review and research gap analysis
  bdc59af docs(stage5): document fine-grained error analysis
  1c06e7d docs(stage6): correct research claims and interpretations
  b683339 docs(stage6): synthesize comprehensive research findings and contributions in STAGE_6_RESEARCH_SYNTHESIS.md
  ```
- **Existing Local Branches**:
  - `main` (active)
  - `backup/pre-reorg-main` (commit `88e674f`: snapshot prior to repository restructuring)

---

## 3. Directory Purposes, Contents, and High-Level Roles

| Directory Path | Tracked? | Files | Size on Disk | Primary Purpose & Functional Role |
|---|:---:|:---:|:---:|---|
| `[ROOT]` | Mixed | 11 | 220 KB | High-level orchestrators (`demo_app.py`, `benchmark_throughput.py`), core documentation (`README.md`, `PROGRESS.md`, `INSIGHTS.md`), and pre-execution audit reports. |
| `MERGED_CSV/` | Gitignored | 63 | 8.87 GB | Raw, full-scale CICIoT2023 dataset captures (`Merged01.csv` through `Merged63.csv`). Preserved for raw replay and offline feature extraction. |
| `data/` | Gitignored | 8 | 177.79 MB | Curated stratified subsample (`sample_stratified.csv`, 529,562 flows) and Stage 2 feature-hash grouped partition assignments. |
| `data/partitions/` | Gitignored | 6 | 65.65 MB | Partition assignment maps, metadata JSONs, and test partitions (`cross_capture_test.csv`, `cross_capture_test_clean.csv`). |
| `src/` | Tracked | 25 | 168 KB | Generation 1 pipeline code (data loader, feature selection, training scripts for Binary, 8-Class, 34-Class, and evaluation). |
| `src_v2/` | Tracked | 22 | 220 KB | Generation 2 leak-free pipeline (hash-partitioning, leak-free preprocessor, 21-model training runner, stage 4 evaluation runner). |
| `experiments/` | Tracked | 7 | 57 KB | Standalone ablation and parameter search experiments (feature pruning, weight calibration, decay sweeps, hierarchical cascades). |
| `models_saved/` | Gitignored | 23 | 274.88 MB | Generation 1 saved model checkpoints and scalers across binary, 8-class, 34-class, and reduced-feature (top-20) tasks. |
| `models_saved_v2/` | Gitignored | 23 | 204.85 MB | Generation 2 verified model checkpoints for all 21 Stage 3 models, plus serialized Strategy 1 and Strategy 2 preprocessors. |
| `results/` | Tracked | 56 | 17.79 MB | Generation 1 evaluation outputs: 30 confusion matrix/ROC figures, 26 benchmark tables (CSV/JSON), and throughput graphs. |
| `results_v2/` | Tracked | 51 | 518 KB | Generation 2 research artifacts: Stage 4-7 markdown reports, per-class evaluation JSONs, stage 4 metrics, and centralized ledger. |
| `notebooks/` | Tracked | 8 | 14.05 MB | Sequential research Jupyter notebooks (01 to 08) covering exploration, phase 1-3 classification, feature reduction, and throughput. |
| `learning/` | Tracked | 2 | 174 KB | Educational module 1 notebook and detailed learning notes on networking and CICIoT2023 fundamentals. |
| `paper/` | Tracked | 10 | 5.66 MB | Academic conference paper package: IEEE LaTeX (`main.tex`), Typst (`paper.typ`), BibTeX (`references.bib`), figures, and compiled PDF. |
| `research/` | Tracked | 19 | 75.29 MB | Literature PDF repository containing 14 peer-reviewed research papers, dataset datasheet, and internal literature surveys. |
| `snuc/` | Gitignored | 26 | 11.93 MB | Academic capstone deliverables for Shiv Nadar University Chennai: Phase 1-3 review presentations, reports, and LaTeX sources. |
| `scratch/` | Gitignored | 1 | 12 KB | Standalone script (`build_notebook.py`) for generating Module 1 interactive notebook. |
| `__pycache__/` | Gitignored | 2 | 66 KB | Compiled Python bytecode for root-level modules. |
| `.venv/` | Gitignored | 36,786 | 1.19 GB | Local Python 3.13 virtual environment with all dependencies installed. |

---

## 4. Full Directory Tree with File Sizes

Below is the complete inventory of all 357 project files (excluding virtual environment internals):

### Directory: `MERGED_CSV` (63 files)
```text
Merged01.csv                                                       147,115,614 bytes
Merged02.csv                                                       154,639,360 bytes
Merged03.csv                                                       144,064,311 bytes
Merged04.csv                                                       139,782,474 bytes
Merged05.csv                                                       153,858,577 bytes
Merged06.csv                                                       148,376,004 bytes
Merged07.csv                                                       145,078,394 bytes
Merged08.csv                                                       147,113,201 bytes
Merged09.csv                                                       140,238,529 bytes
Merged10.csv                                                       189,279,680 bytes
Merged11.csv                                                       187,937,973 bytes
Merged12.csv                                                       185,310,837 bytes
Merged13.csv                                                       146,677,937 bytes
Merged14.csv                                                       186,540,311 bytes
Merged15.csv                                                       148,412,810 bytes
Merged16.csv                                                       158,106,214 bytes
Merged17.csv                                                       151,106,632 bytes
Merged18.csv                                                       147,513,362 bytes
Merged19.csv                                                       145,027,913 bytes
Merged20.csv                                                       148,686,193 bytes
Merged21.csv                                                       143,085,327 bytes
Merged22.csv                                                       152,731,221 bytes
Merged23.csv                                                       142,220,251 bytes
Merged24.csv                                                       153,542,038 bytes
Merged25.csv                                                       144,688,091 bytes
Merged26.csv                                                       181,572,383 bytes
Merged27.csv                                                       148,887,180 bytes
Merged28.csv                                                       145,934,611 bytes
Merged29.csv                                                       148,223,435 bytes
Merged30.csv                                                       143,719,665 bytes
Merged31.csv                                                       141,004,713 bytes
Merged32.csv                                                       145,464,744 bytes
Merged33.csv                                                       163,994,297 bytes
Merged34.csv                                                       185,731,135 bytes
Merged35.csv                                                       190,164,566 bytes
Merged36.csv                                                       153,564,144 bytes
Merged37.csv                                                       142,332,741 bytes
Merged38.csv                                                       150,717,367 bytes
Merged39.csv                                                       139,874,681 bytes
Merged40.csv                                                       143,915,067 bytes
Merged41.csv                                                       147,277,898 bytes
Merged42.csv                                                       180,077,709 bytes
Merged43.csv                                                       143,017,588 bytes
Merged44.csv                                                       149,972,599 bytes
Merged45.csv                                                       139,154,007 bytes
Merged46.csv                                                       140,142,359 bytes
Merged47.csv                                                       135,841,629 bytes
Merged48.csv                                                       135,841,629 bytes
Merged49.csv                                                       132,769,679 bytes
Merged50.csv                                                       124,168,219 bytes
Merged51.csv                                                        44,297,519 bytes
Merged52.csv                                                        13,578,019 bytes
Merged53.csv                                                       187,394,155 bytes
Merged54.csv                                                       150,268,183 bytes
Merged55.csv                                                       155,909,259 bytes
Merged56.csv                                                       153,672,571 bytes
Merged57.csv                                                       178,722,424 bytes
Merged58.csv                                                       146,595,068 bytes
Merged59.csv                                                       141,968,337 bytes
Merged60.csv                                                       142,876,398 bytes
Merged61.csv                                                       146,983,288 bytes
Merged62.csv                                                       144,903,857 bytes
Merged63.csv                                                        88,473,082 bytes
```

### Directory: `[ROOT]` (11 files)
```text
.gitignore                                                                 443 bytes
INSIGHTS.md                                                             22,892 bytes
PROGRESS.md                                                             18,008 bytes
PROJECT_ORIENTATION_REPORT.md                                           31,105 bytes
README.md                                                               14,487 bytes
RESEARCH_PAPERS_CRITICAL_ANALYSIS.md                                    34,665 bytes
STAGE_3_CANARY_REPORT.md                                                10,750 bytes
STAGE_3_METHODOLOGY_REVIEW.md                                           32,240 bytes
benchmark_throughput.py                                                 22,890 bytes
demo_app.py                                                             32,318 bytes
requirements.txt                                                           342 bytes
```

### Directory: `__pycache__` (2 files)
```text
benchmark_throughput.cpython-313.pyc                                    29,514 bytes
demo_app.cpython-313.pyc                                                37,154 bytes
```

### Directory: `data` (1 files)
```text
sample_stratified.csv                                              111,598,532 bytes
```

### Directory: `data\partitions` (5 files)
```text
cross_capture_metadata.json                                              1,825 bytes
cross_capture_test.csv                                              13,399,286 bytes
cross_capture_test_clean.csv                                        12,158,080 bytes
partition_assignments.csv                                           25,089,011 bytes
partition_metadata.json                                                  5,312 bytes
```

### Directory: `data\partitions\ablation_strategy2` (2 files)
```text
metadata.json                                                            1,706 bytes
partition_assignments.csv                                           24,170,492 bytes
```

### Directory: `experiments` (7 files)
```text
feature_decay_sweep.py                                                   7,405 bytes
feature_pruning.py                                                       7,019 bytes
hierarchical_cascade.py                                                 11,082 bytes
hyperparameter_tuning.py                                                 6,983 bytes
natural_distribution_evaluation.py                                       8,627 bytes
uncapped_pure34_benchmark.py                                             9,568 bytes
weight_calibration.py                                                    6,817 bytes
```

### Directory: `learning\module_1` (2 files)
```text
01_networking_and_ciciot2023_fundamentals.ipynb                        129,210 bytes
MODULE_1_LEARNING_NOTES.md                                              44,755 bytes
```

### Directory: `models_saved\attack_34class` (6 files)
```text
attack34_scaler.joblib                                                   1,503 bytes
lightgbm.joblib                                                      9,499,844 bytes
logistic_regression.joblib                                               7,093 bytes
pytorch_dnn.pt                                                          71,243 bytes
random_forest.joblib                                                97,292,745 bytes
xgboost.joblib                                                       5,086,562 bytes
```

### Directory: `models_saved\binary` (6 files)
```text
binary_scaler.joblib                                                     1,503 bytes
lightgbm.joblib                                                        361,252 bytes
logistic_regression.joblib                                               1,301 bytes
pytorch_dnn.pt                                                          62,923 bytes
random_forest.joblib                                                12,877,417 bytes
xgboost.joblib                                                         248,390 bytes
```

### Directory: `models_saved\category_8class` (6 files)
```text
category8_scaler.joblib                                                  1,503 bytes
lightgbm.joblib                                                      2,818,532 bytes
logistic_regression.joblib                                               2,501 bytes
pytorch_dnn.pt                                                          64,459 bytes
random_forest.joblib                                                38,905,497 bytes
xgboost.joblib                                                       1,983,893 bytes
```

### Directory: `models_saved\reduced_features` (5 files)
```text
lightgbm_top20.joblib                                                9,482,564 bytes
random_forest_top20.joblib                                         104,010,217 bytes
reduced_20_scaler.joblib                                                 1,063 bytes
top_20_feature_names.joblib                                                240 bytes
xgboost_top20.joblib                                                 5,446,154 bytes
```

### Directory: `models_saved_v2` (23 files)
```text
ablation_s2_binary_lightgbm.joblib                                     361,028 bytes
ablation_s2_category_lightgbm.joblib                                 2,813,828 bytes
ablation_s2_fine_grained_lightgbm.joblib                            11,883,236 bytes
ablation_s2_fine_grained_xgboost.joblib                              8,691,871 bytes
ablation_unweighted_fine_grained_lightgbm.joblib                    11,877,604 bytes
ablation_unweighted_fine_grained_xgboost.joblib                      8,392,709 bytes
binary_lightgbm.joblib                                                 360,900 bytes
binary_logistic_regression.joblib                                        1,301 bytes
binary_pytorch_dnn_torch.pt                                             63,337 bytes
binary_random_forest.joblib                                         12,350,473 bytes
binary_xgboost.joblib                                                  256,642 bytes
category_lightgbm.joblib                                             2,813,812 bytes
category_logistic_regression.joblib                                      2,501 bytes
category_pytorch_dnn_torch.pt                                           64,917 bytes
category_random_forest.joblib                                       36,468,665 bytes
category_xgboost.joblib                                              2,109,131 bytes
fine_grained_lightgbm.joblib                                        11,881,508 bytes
fine_grained_logistic_regression.joblib                                  7,093 bytes
fine_grained_pytorch_dnn_torch.pt                                       71,789 bytes
fine_grained_random_forest.joblib                                   95,653,065 bytes
fine_grained_xgboost.joblib                                          8,667,184 bytes
preprocessor_strategy1.joblib                                            2,446 bytes
preprocessor_strategy2.joblib                                            2,446 bytes
```

### Directory: `notebooks` (8 files)
```text
01_data_exploration_sampling.ipynb                                     235,231 bytes
02_phase1_binary_classification.ipynb                                2,309,065 bytes
03_phase2_eight_class_classification.ipynb                           2,389,600 bytes
04_phase3_thirtyfour_class_classification.ipynb                      6,734,853 bytes
05_feature_selection_latency.ipynb                                     996,531 bytes
06_capstone_synthesis_presentation.ipynb                             1,392,937 bytes
07_uncapped_high_accuracy_benchmark.ipynb                              464,612 bytes
08_wire_speed_throughput_benchmark.ipynb                               211,084 bytes
```

### Directory: `paper` (5 files)
```text
README.md                                                                2,137 bytes
main.tex                                                                16,592 bytes
paper.typ                                                               14,935 bytes
references.bib                                                           3,085 bytes
research_paper.pdf                                                   1,469,090 bytes
```

### Directory: `paper\figures` (5 files)
```text
attack34_cm_xgboost.png                                              2,222,144 bytes
capstone_master_synthesis.png                                        1,032,022 bytes
feature_importance_ranking.png                                         434,048 bytes
feature_reduction_comparison.png                                       234,243 bytes
iot_defense_architecture.png                                           511,578 bytes
```

### Directory: `research` (19 files)
```text
A resource efficient IoT intrusion detection model using hybrid feature selection for edge computing.pdf    6,202,571 bytes
AlHejriAlTam_ImprovingIoTSecurityExplainableHybridCNNTransformerFederatedLearning.pdf    3,500,787 bytes
An optimized gradient boosting framework for IoT intrusion detection - a comprehensive evaluation on the CICIoT2023 dataset.pdf    3,224,364 bytes
BooijChiscop_ToNIoTHeterogeneityStandardizationIoTIntrusionDatasets.pdf    4,195,679 bytes
CICIoT2023 A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environment.pdf    2,820,040 bytes
CICIoT2023 Capstone Research Analysis_260830_210326.pdf                215,995 bytes
CICIoT2023 Model Pipeline Analysis_260830_210234.pdf                   174,554 bytes
ChoudharyTanwar_DatasetGenerationEvaluationIoTIntrusionDetection.pdf   20,070,003 bytes
Comparative analysis of deep learning and traditional methods for IoT botnet detection using a multi-model framework across diverse datasets.pdf    6,980,001 bytes
Dataset.pdf                                                          3,819,227 bytes
EnhancingnetworksecurityinindustrialIoTenvironmentsaDeepCLG.pdf      3,394,270 bytes
Feature-Efficient_Machine_Learning_Framework_for_Binary_and_Multiclass_Intrusion_Detection_in_IoT_Networks.pdf    1,141,429 bytes
FenanirSemchedine_SmartIntrusionDetectionIoTEdgeComputingUsingFederatedLearning.pdf    1,585,020 bytes
IoT Edge IDS Optimization Research Original.pdf                        491,992 bytes
Khan_Alam_ResourceEfficientIoTIntrusionDetectionHybridFeatureSelection.pdf    6,202,571 bytes
MoussaouiKmiti_ResilientFederatedIntrusionDetectionExplainableAIRobustCNNLSTMArchitectureExtremeNonIIDDataDistributions.pdf    5,276,637 bytes
Moustafa_DistributedArchitectureAIBasedSecurityEdgeTONIoTDatasets.pdf      854,253 bytes
QaddosYaseen_NovelIntrusionDetectionFrameworkOptimizingIoTSecurity.pdf    3,650,203 bytes
UmarYasmeen_EnergyEfficientDeepLearningBasedIntrusionDetectionSystemEdgeComputingNovelDNNKDQModel.pdf    5,152,445 bytes
```

### Directory: `results\figures` (30 files)
```text
attack34_cm_lightgbm.png                                             2,254,299 bytes
attack34_cm_logistic_regression.png                                  2,283,203 bytes
attack34_cm_pytorch_dnn.png                                          2,268,716 bytes
attack34_cm_random_forest.png                                        2,245,586 bytes
attack34_cm_xgboost.png                                              2,222,144 bytes
binary_cm_lightgbm.png                                                 106,723 bytes
binary_cm_logistic_regression.png                                      107,262 bytes
binary_cm_pytorch_dnn.png                                              106,573 bytes
binary_cm_random_forest.png                                            106,806 bytes
binary_cm_xgboost.png                                                  105,499 bytes
binary_roc_pr_lightgbm.png                                             219,574 bytes
binary_roc_pr_logistic_regression.png                                  233,331 bytes
binary_roc_pr_pytorch_dnn.png                                          227,792 bytes
binary_roc_pr_random_forest.png                                        224,129 bytes
binary_roc_pr_xgboost.png                                              223,395 bytes
capstone_master_synthesis.png                                        1,032,022 bytes
category8_cm_lightgbm.png                                              335,815 bytes
category8_cm_logistic_regression.png                                   357,580 bytes
category8_cm_pytorch_dnn.png                                           345,975 bytes
category8_cm_random_forest.png                                         335,695 bytes
category8_cm_xgboost.png                                               342,929 bytes
feature_decay_curve.png                                                202,642 bytes
feature_importance_ranking.png                                         434,048 bytes
feature_reduction_comparison.png                                       234,243 bytes
iot_defense_architecture.png                                           511,578 bytes
throughput_benchmark.png                                               319,439 bytes
uncapped_binary_curves.png                                             271,139 bytes
uncapped_multiclass_matrix.png                                         401,686 bytes
uncapped_natural_distribution.png                                      216,589 bytes
uncapped_vs_capped_paradox.png                                         310,981 bytes
```

### Directory: `results\tables` (26 files)
```text
attack34_minority_attacks_benchmark.csv                                  2,178 bytes
attack34_models_benchmark.csv                                              859 bytes
attack34_models_benchmark.json                                           1,898 bytes
attack34_per_class_reports.json                                         30,241 bytes
binary_models_benchmark.csv                                                951 bytes
binary_models_benchmark.json                                             2,039 bytes
capstone_executive_trajectory.csv                                          286 bytes
capstone_master_comparison.csv                                           1,357 bytes
capstone_master_comparison.json                                          4,717 bytes
category8_models_benchmark.csv                                             862 bytes
category8_models_benchmark.json                                          1,901 bytes
category8_per_class_reports.json                                         8,509 bytes
exp1_weight_calibration.csv                                                720 bytes
exp2_hyperparameter_tuning.csv                                             687 bytes
exp3_feature_pruning.csv                                                   709 bytes
exp4_hierarchical_cascade.csv                                              358 bytes
exp5_distribution_reconciliation.csv                                       456 bytes
exp7_uncapped_pure34.csv                                                   543 bytes
feature_decay_sweep.csv                                                  2,514 bytes
feature_importance_rankings.csv                                          4,127 bytes
feature_reduction_benchmark.csv                                            620 bytes
feature_reduction_benchmark.json                                         2,009 bytes
throughput_benchmark.csv                                                 1,811 bytes
uncapped_binary_benchmark.csv                                              354 bytes
uncapped_unified7_classification_report.csv                                684 bytes
uncapped_vs_balanced_reconciliation.csv                                    620 bytes
```

### Directory: `results_v2` (4 files)
```text
STAGE_4_EVALUATION_REPORT.md                                            23,248 bytes
STAGE_5_ERROR_ANALYSIS.md                                               14,111 bytes
STAGE_6_RESEARCH_SYNTHESIS.md                                           26,983 bytes
STAGE_7_LITERATURE_REVIEW.md                                            48,261 bytes
```

### Directory: `results_v2\metrics` (3 files)
```text
binary_pytorch_dnn_dnn_loss.json                                           189 bytes
category_pytorch_dnn_dnn_loss.json                                         187 bytes
fine_grained_pytorch_dnn_dnn_loss.json                                     185 bytes
```

### Directory: `results_v2\metrics\per_class` (21 files)
```text
ablation_s2_binary_lightgbm_per_class.json                               1,551 bytes
ablation_s2_category_lightgbm_per_class.json                             3,531 bytes
ablation_s2_fine_grained_lightgbm_per_class.json                        13,566 bytes
ablation_s2_fine_grained_xgboost_per_class.json                         13,525 bytes
ablation_unweighted_fine_grained_lightgbm_per_class.json                18,661 bytes
ablation_unweighted_fine_grained_xgboost_per_class.json                 18,536 bytes
binary_lightgbm_per_class.json                                           2,180 bytes
binary_logistic_regression_per_class.json                                2,217 bytes
binary_pytorch_dnn_per_class.json                                        2,181 bytes
binary_random_forest_per_class.json                                      2,253 bytes
binary_xgboost_per_class.json                                            2,254 bytes
category_lightgbm_per_class.json                                         5,014 bytes
category_logistic_regression_per_class.json                              5,016 bytes
category_pytorch_dnn_per_class.json                                      5,014 bytes
category_random_forest_per_class.json                                    5,008 bytes
category_xgboost_per_class.json                                          5,020 bytes
fine_grained_lightgbm_per_class.json                                    18,524 bytes
fine_grained_logistic_regression_per_class.json                         18,656 bytes
fine_grained_pytorch_dnn_per_class.json                                 18,676 bytes
fine_grained_random_forest_per_class.json                               18,539 bytes
fine_grained_xgboost_per_class.json                                     18,544 bytes
```

### Directory: `results_v2\metrics\stage4` (21 files)
```text
ablation_s2_binary_lightgbm_stage4.json                                  2,664 bytes
ablation_s2_category_lightgbm_stage4.json                                4,792 bytes
ablation_s2_fine_grained_lightgbm_stage4.json                           16,284 bytes
ablation_s2_fine_grained_xgboost_stage4.json                            16,288 bytes
ablation_unweighted_fine_grained_lightgbm_stage4.json                   16,330 bytes
ablation_unweighted_fine_grained_xgboost_stage4.json                    16,255 bytes
binary_lightgbm_stage4.json                                              2,646 bytes
binary_logistic_regression_stage4.json                                   2,667 bytes
binary_pytorch_dnn_stage4.json                                           2,652 bytes
binary_random_forest_stage4.json                                         2,653 bytes
binary_xgboost_stage4.json                                               2,640 bytes
category_lightgbm_stage4.json                                            4,770 bytes
category_logistic_regression_stage4.json                                 4,783 bytes
category_pytorch_dnn_stage4.json                                         4,782 bytes
category_random_forest_stage4.json                                       4,776 bytes
category_xgboost_stage4.json                                             4,764 bytes
fine_grained_lightgbm_stage4.json                                       16,290 bytes
fine_grained_logistic_regression_stage4.json                            16,343 bytes
fine_grained_pytorch_dnn_stage4.json                                    16,322 bytes
fine_grained_random_forest_stage4.json                                  16,318 bytes
fine_grained_xgboost_stage4.json                                        16,257 bytes
```

### Directory: `results_v2\tables` (2 files)
```text
centralized_experiment_ledger.csv                                        8,591 bytes
stage4_test_results.csv                                                  6,644 bytes
```

### Directory: `scratch` (1 files)
```text
build_notebook.py                                                       12,884 bytes
```

### Directory: `snuc` (26 files)
```text
Archon_PPT_Ph1Re2.pdf                                                  487,643 bytes
Archon_PPT_Ph1Re2.pptx                                               1,233,661 bytes
Archon_PPT_Ph1Re3.pdf                                                1,086,808 bytes
Archon_PPT_Ph1Re3.pptx                                               2,663,173 bytes
Archon_Report_Ph1Re2.pdf                                               277,254 bytes
Archon_Report_Ph1Re2_(Non-humanize).pdf                                177,296 bytes
Archon_Report_Ph1Re3.pdf                                               850,405 bytes
CICIoT2023 Capstone Research Analysis_260830_210326.pdf                215,995 bytes
CICIoT2023 Model Pipeline Analysis_260830_210234.pdf                   174,554 bytes
IoT Edge IDS Optimization Research Original.pdf                        491,992 bytes
IoT Edge IDS Optimization Research Presentation Content.pdf            290,219 bytes
IoT_IDS_Capstone-v3.pptx                                             1,169,442 bytes
IoT_IDS_Capstone_SNUC_Template.pdf                                     303,618 bytes
IoT_IDS_Capstone_SNUC_Template.pptx                                    768,241 bytes
PROJECT_REVIEW_2.md                                                     28,698 bytes
PROJECT_REVIEW_2_(Humanized).tex                                        20,873 bytes
RESEARCH_PAPERS_CRITICAL_ANALYSIS.md                                    34,665 bytes
REVIEW_3_PPT_CONTENT.md                                                 12,688 bytes
Research_paper_survey.pdf                                              375,480 bytes
Research_paper_survey.pptx                                             784,529 bytes
Research_paper_survey_(with metrics).pptx                               92,870 bytes
Review-2 front-page.pdf                                                 79,614 bytes
Review-3 front-page.pdf                                                 88,938 bytes
review-3_ppt.pdf                                                       755,477 bytes
review_2.tex                                                            20,891 bytes
review_3.tex                                                            24,301 bytes
```

### Directory: `src` (12 files)
```text
__init__.py                                                                129 bytes
capstone_synthesis.py                                                    9,143 bytes
config.py                                                                4,876 bytes
create_sample.py                                                         5,083 bytes
data_loader.py                                                           2,674 bytes
evaluate.py                                                              7,217 bytes
feature_selection.py                                                    11,767 bytes
models.py                                                                7,550 bytes
preprocessing.py                                                         4,857 bytes
train_attack_34class.py                                                  8,255 bytes
train_binary.py                                                          5,999 bytes
train_category_8class.py                                                 6,253 bytes
```

### Directory: `src\__pycache__` (13 files)
```text
__init__.cpython-313.pyc                                                   251 bytes
capstone_synthesis.cpython-313.pyc                                      10,008 bytes
config.cpython-313.pyc                                                   4,237 bytes
create_sample.cpython-313.pyc                                            6,507 bytes
data_loader.cpython-313.pyc                                              3,614 bytes
evaluate.cpython-313.pyc                                                 9,097 bytes
feature_decay_sweep.cpython-313.pyc                                      7,370 bytes
feature_selection.cpython-313.pyc                                       14,811 bytes
models.cpython-313.pyc                                                  10,082 bytes
preprocessing.cpython-313.pyc                                            6,309 bytes
train_attack_34class.cpython-313.pyc                                     8,796 bytes
train_binary.cpython-313.pyc                                             6,450 bytes
train_category_8class.cpython-313.pyc                                    6,952 bytes
```

### Directory: `src_v2` (11 files)
```text
__init__.py                                                                168 bytes
ablation_strategy2.py                                                    6,252 bytes
config.py                                                                5,873 bytes
cross_capture.py                                                         3,522 bytes
data_partitioning.py                                                     7,916 bytes
evaluate.py                                                              8,465 bytes
models.py                                                               10,292 bytes
preprocessing.py                                                         8,689 bytes
run_stage3.py                                                           23,210 bytes
run_stage4_eval.py                                                      21,471 bytes
verify_partitions.py                                                     6,362 bytes
```

### Directory: `src_v2\__pycache__` (11 files)
```text
__init__.cpython-313.pyc                                                   292 bytes
ablation_strategy2.cpython-313.pyc                                       8,333 bytes
config.cpython-313.pyc                                                   4,912 bytes
cross_capture.cpython-313.pyc                                            4,862 bytes
data_partitioning.cpython-313.pyc                                        9,558 bytes
evaluate.cpython-313.pyc                                                 9,714 bytes
models.cpython-313.pyc                                                  13,542 bytes
preprocessing.cpython-313.pyc                                           11,507 bytes
run_stage3.cpython-313.pyc                                              20,663 bytes
run_stage4_eval.cpython-313.pyc                                         25,405 bytes
verify_partitions.cpython-313.pyc                                        9,329 bytes
```

---

## 5. Critical Files and Artifacts Breakdown

### 5.1 Datasets and Partitions (`data/`, `MERGED_CSV/`)
- `MERGED_CSV/Merged01.csv` through `Merged63.csv`: Raw, uncompressed network captures from the CICIoT2023 testbed totaling 8.87 GB.
- `data/sample_stratified.csv` (111.6 MB, 529,562 rows): Curated stratified sample with a 20,000 packet cap on volumetric floods, preserving 100% of stealthy minority vectors.
- `data/partitions/partition_assignments.csv` (25.1 MB): Hash-grouped assignment table assigning every row in `sample_stratified.csv` to Train (350,821), Val (75,172), or Test (75,105) with 0% feature overlap.
- `data/partitions/ablation_strategy2/partition_assignments.csv` (24.2 MB): Strategy 2 assignment table purging all 8,740 conflicting collision hashes (Train: 337,838, Val: 72,390, Test: 72,454).
- `data/partitions/cross_capture_test_clean.csv` (12.2 MB, 57,280 rows): Disjoint temporal test partition sampled from subsequent sessions (`Merged26.csv`–`Merged40.csv`), purged of training collision hashes.

### 5.2 Source Code and Pipelines (`src/`, `src_v2/`)
- **`src_v2/data_partitioning.py`**: Stage 2 engine creating atomic feature-hash groupings and mutually disjoint partitions.
- **`src_v2/preprocessing.py`**: Implements `LeakFreePreprocessor`, guaranteeing that finite-max capping and z-score scaling parameters are learned strictly on training partitions.
- **`src_v2/models.py`**: Model architecture factory providing SGD, Random Forest, LightGBM, XGBoost, and PyTorch Tabular DNN with cost-sensitive class weighting.
- **`src_v2/run_stage3.py`**: Training orchestrator that fits, audits RAM/duration, validates, and serializes all 21 Stage 3 model checkpoints.
- **`src_v2/run_stage4_eval.py`**: Final read-only test evaluation runner across internal test and cross-capture test partitions.
- **`src_v2/verify_partitions.py`**: Mathematical verification script proving zero feature overlap and partition consistency.

### 5.3 Model Checkpoints (`models_saved_v2/`)
- **Serialized Preprocessors**: `preprocessor_strategy1.joblib` (350,821 samples) and `preprocessor_strategy2.joblib` (337,838 samples).
- **21 Stage 3 Checkpoints**: Spanning Phase 1 (Binary), Phase 2 (Category 8-class), Phase 3 (Fine-grained 34-class), Phase 4 (Strategy 2 Collision-Purged), and Phase 5 (Unweighted Ablation).

### 5.4 Research Reports (`results_v2/`, `[ROOT]`)
- **`results_v2/STAGE_4_EVALUATION_REPORT.md`**: Master test evaluation across internal and same-testbed cross-capture sets.
- **`results_v2/STAGE_5_ERROR_ANALYSIS.md`**: Fine-grained error analysis, worst-10 attack classes, and confusion patterns.
- **`results_v2/STAGE_6_RESEARCH_SYNTHESIS.md`**: Contribution synthesis, research claims, limitations, and methodological reconciliation.
- **`results_v2/STAGE_7_LITERATURE_REVIEW.md`**: Literature review and gap analysis across 14 peer-reviewed research papers.
- **`results_v2/tables/centralized_experiment_ledger.csv`**: Centralized benchmark ledger logging validation metrics, latency, memory, and throughput for all 21 models.
- **`results_v2/tables/stage4_test_results.csv`**: Master consolidated test evaluation table across partitions.
- **`STAGE_3_METHODOLOGY_REVIEW.md`** & **`STAGE_3_CANARY_REPORT.md`**: Root-level methodological audit and canary execution report.

---

## 6. Duplicates, Misplaced Files, Obsolete Artifacts, and Inconsistencies

### 6.1 Exact File Duplicates (10 Groups Identified)
A byte-level SHA-256 hash comparison across all project files identified 10 duplicate groups totaling over 11 MB of redundant storage:

1. **SHA-256 `97d958f40c5b...` (6,202,571 bytes)**:
   - `research\A resource efficient IoT intrusion detection model using hybrid feature selection for edge computing.pdf`
   - `research\Khan_Alam_ResourceEfficientIoTIntrusionDetectionHybridFeatureSelection.pdf`
1. **SHA-256 `acbf49ccf738...` (2,222,144 bytes)**:
   - `paper\figures\attack34_cm_xgboost.png`
   - `results\figures\attack34_cm_xgboost.png`
1. **SHA-256 `e12a3faad6b3...` (1,032,022 bytes)**:
   - `paper\figures\capstone_master_synthesis.png`
   - `results\figures\capstone_master_synthesis.png`
1. **SHA-256 `8e12b4046423...` (511,578 bytes)**:
   - `paper\figures\iot_defense_architecture.png`
   - `results\figures\iot_defense_architecture.png`
1. **SHA-256 `410ad764aef8...` (491,992 bytes)**:
   - `research\IoT Edge IDS Optimization Research Original.pdf`
   - `snuc\IoT Edge IDS Optimization Research Original.pdf`
1. **SHA-256 `9b98915b0e3c...` (434,048 bytes)**:
   - `paper\figures\feature_importance_ranking.png`
   - `results\figures\feature_importance_ranking.png`
1. **SHA-256 `7e5545ca2932...` (234,243 bytes)**:
   - `paper\figures\feature_reduction_comparison.png`
   - `results\figures\feature_reduction_comparison.png`
1. **SHA-256 `a3d31b66fdf3...` (215,995 bytes)**:
   - `research\CICIoT2023 Capstone Research Analysis_260830_210326.pdf`
   - `snuc\CICIoT2023 Capstone Research Analysis_260830_210326.pdf`
1. **SHA-256 `c71787a6d98c...` (174,554 bytes)**:
   - `research\CICIoT2023 Model Pipeline Analysis_260830_210234.pdf`
   - `snuc\CICIoT2023 Model Pipeline Analysis_260830_210234.pdf`
1. **SHA-256 `fb45fa24430f...` (34,665 bytes)**:
   - `RESEARCH_PAPERS_CRITICAL_ANALYSIS.md`
   - `snuc\RESEARCH_PAPERS_CRITICAL_ANALYSIS.md`

### 6.2 Misplaced and Redundant Documentation Files
1. **`RESEARCH_PAPERS_CRITICAL_ANALYSIS.md`**: Present in both repository root (`[ROOT]`) and `snuc/`. Both copies are completely identical (34,665 bytes).
2. **Duplicate Research PDFs in `research/` and `snuc/`**:
   - `research/IoT Edge IDS Optimization Research Original.pdf` is identical to `snuc/IoT Edge IDS Optimization Research Original.pdf` (491,992 bytes).
   - `research/CICIoT2023 Capstone Research Analysis_260830_210326.pdf` is identical to `snuc/CICIoT2023 Capstone Research Analysis_260830_210326.pdf` (215,995 bytes).
   - `research/CICIoT2023 Model Pipeline Analysis_260830_210234.pdf` is identical to `snuc/CICIoT2023 Model Pipeline Analysis_260830_210234.pdf` (174,554 bytes).
   - `research/` contains two identical copies of the Khan & Alam paper under different names: `A resource efficient IoT intrusion detection model...pdf` and `Khan_Alam_ResourceEfficient...pdf` (6,202,571 bytes).
3. **Misplaced Generation Script**: `scratch/build_notebook.py` contains hardcoded relative paths (`../../data/sample_stratified.csv`) indicating it was written to run from `learning/module_1/` rather than `scratch/`.

### 6.3 Stage Report Location Inconsistencies
- Stages 4, 5, 6, and 7 reports are stored in `results_v2/` (`STAGE_4_EVALUATION_REPORT.md`, `STAGE_5_ERROR_ANALYSIS.md`, `STAGE_6_RESEARCH_SYNTHESIS.md`, `STAGE_7_LITERATURE_REVIEW.md`).
- Stage 3 reports are stored at the repository root (`STAGE_3_METHODOLOGY_REVIEW.md`, `STAGE_3_CANARY_REPORT.md`).
- Stages 1 and 2 reports (`STAGE_1_AUDIT_REPORT.md`, `STAGE_2_INVESTIGATION_REPORT.md`) reside in the external agent brain artifacts directory (`<appDataDir>/brain/<conversation-id>/`) and were not committed to `results_v2/`.

### 6.4 Obsolete / Legacy Artifacts vs Modern v2 Artifacts
- **`src/` vs `src_v2/`**: `src/` represents Generation 1 exploratory scripts (unpartitioned random splitting). `src_v2/` is the active, verified, leak-free pipeline.
- **`models_saved/` vs `models_saved_v2/`**: `models_saved/` holds legacy models trained with random 80/20 splitting. `models_saved_v2/` contains the audited Stage 3 checkpoints.
- **`results/` vs `results_v2/`**: `results/` contains exploratory tables and figures from Generation 1. `results_v2/` contains the definitive research outputs, per-class metrics, and centralized ledger.
- **Empty Directory**: `results_v2/figures/` exists as a directory but currently contains no generated figure files.

---

## 7. Inter-Directory Dependencies and Cross-References

The project exhibits clear architectural coupling across its components:

```mermaid
flowchart TD
    subgraph Data Layer
        RAW["MERGED_CSV (63 raw captures)"] --> SAMPLE["data/sample_stratified.csv (529k flows)"]
        SAMPLE --> PART["data/partitions (Hash-Grouped S1, S2, Cross-Capture)"]
    end

    subgraph Pipeline v2 (Active Research)
        PART --> PRE2["src_v2/preprocessing.py"]
        PRE2 --> RUN3["src_v2/run_stage3.py"]
        MOD2["src_v2/models.py"] --> RUN3
        RUN3 --> CKPT2["models_saved_v2/ (21 Checkpoints + 2 Preprocessors)"]
        CKPT2 --> RUN4["src_v2/run_stage4_eval.py"]
        RUN4 --> RES2["results_v2/ (Stage 4-7 Reports, Ledger, Metrics)"]
    end

    subgraph Legacy Pipeline v1
        SAMPLE --> SRC1["src/ (train_binary, train_category, train_34class)"]
        SRC1 --> CKPT1["models_saved/ (v1 models)"]
        CKPT1 --> RES1["results/ (v1 figures & tables)"]
        RES1 --> DEMO["demo_app.py (Streamlit GUI)"]
        RES1 --> BENCH["benchmark_throughput.py"]
    end

    subgraph Academic Deliverables
        RES1 --> PAPER["paper/ (IEEE Conference Paper)"]
        RES2 --> SNUC["snuc/ (Reviews 1-3 Presentations & Reports)"]
        RESEARCH["research/ (14 Research PDFs)"] --> RES2
    end
```

### Key Dependency Details:
1. **`demo_app.py`**: Depends directly on `models_saved/` (loads `models_saved/binary/xgboost.joblib`, `models_saved/category_8class/xgboost.joblib`, `models_saved/attack_34class/xgboost.joblib`, and corresponding scalers) and `data/sample_stratified.csv`.
2. **`benchmark_throughput.py`**: Evaluates models from `models_saved/` and outputs benchmark figures and tables to `results/figures/` and `results/tables/`.
3. **`experiments/`**: Most experiment scripts (`feature_decay_sweep.py`, `feature_pruning.py`, `hierarchical_cascade.py`) import configuration and class definitions from `src.config` and load `data/sample_stratified.csv`.
4. **`notebooks/`**: Notebooks 01–06 load from `src/` and output to `results/`.
5. **`paper/`**: Embeds figure files (`attack34_cm_xgboost.png`, `capstone_master_synthesis.png`, `feature_importance_ranking.png`, `feature_reduction_comparison.png`, `iot_defense_architecture.png`) which are exact byte-level duplicates of files in `results/figures/`.
6. **`src_v2/`**: Completely self-contained modular architecture. `src_v2/run_stage3.py` and `src_v2/run_stage4_eval.py` import strictly from `src_v2.config`, `src_v2.preprocessing`, `src_v2.models`, and `src_v2.evaluate`.

---

## 8. Summary & Recommendations

1. **Integrity Confirmation**: The repository is in a healthy, reproducible state. All 21 Stage 3 model checkpoints in `models_saved_v2/` reproduce their exact validation metrics, and Stage 4 evaluation metrics in `results_v2/tables/stage4_test_results.csv` are fully accounted for.
2. **Disk Footprint**: Over 85% of repository disk usage (8.87 GB) resides in `MERGED_CSV/`, which is properly gitignored.
3. **Opportunities for Future Housekeeping (Post-Audit)**:
   - Remove duplicate research PDFs in `research/` and `snuc/` (saving ~11 MB).
   - Remove the duplicate copy of `RESEARCH_PAPERS_CRITICAL_ANALYSIS.md` from either root or `snuc/`.
   - Consolidate Stage 1 and Stage 2 reports into `results_v2/` alongside Stages 4–7 for uniform documentation provenance.
   - Move `STAGE_3_CANARY_REPORT.md` and `STAGE_3_METHODOLOGY_REVIEW.md` into `results_v2/` to keep the root directory uncluttered.
