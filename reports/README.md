# Research Reports & Methodological Audits (Stages 1–7)

This directory contains the consolidated, chronological research reports and empirical audits produced across the capstone project investigating Machine Learning and Deep Learning baselines for IoT Network Intrusion Detection on the **CICIoT2023** benchmark.

---

## 1. Master Research Trajectory

| Stage | Document | Scope & Methodology | Primary Finding / Calibrated Result |
|:---:|---|---|---|
| **Stage 1** | [`STAGE_1_AUDIT_REPORT.md`](STAGE_1_AUDIT_REPORT.md) | Forensic dataset audit of 529,562 flows from `sample_stratified.csv`. Quantified exact duplicates, identical feature vectors, and label collisions. | Found 8,740 unique feature vectors mapped to conflicting labels ($N=54,828$ flows), with $48.2\%$ involving `BenignTraffic`. Proved naive random splitting causes data leakage. |
| **Stage 2** | [`STAGE_2_INVESTIGATION_REPORT.md`](STAGE_2_INVESTIGATION_REPORT.md) | Architectural design of feature-hash grouping, collision purging (Strategy 2), and disjoint temporal cross-capture partition (`Merged26`–`Merged40`). | Proved 0% feature-vector leakage across splits. Constructed clean temporal test set ($N=57,280$) purged of all training collision hashes. |
| **Stage 3** | [`STAGE_3_METHODOLOGY_REVIEW.md`](STAGE_3_METHODOLOGY_REVIEW.md) | Pre-execution audit of preprocessing parity, finite-max capping, cost-sensitive class weights, and training configurations for 21 models. | Reconciled DNN budget (5 epochs), verified unscaled trees vs. $z$-score scaled SGD/DNN, and confirmed zero test-data exposure. |
| **Stage 3** | [`STAGE_3_CANARY_REPORT.md`](STAGE_3_CANARY_REPORT.md) | Canary execution and verification of Binary SGD, Random Forest, and LightGBM prior to full suite training. | Validated training runtime, memory ceiling ($\le 8$ GB), and mathematical metric computation parity. |
| **Stage 4** | [`STAGE_4_EVALUATION_REPORT.md`](STAGE_4_EVALUATION_REPORT.md) | Read-only evaluation of all 21 Stage 3 model checkpoints on Internal Test ($N=75,105$) and same-testbed Temporal Cross-Capture ($N=57,280$). | Random Forest led binary internal test Macro-F1 ($0.7690$); LightGBM led Category ($0.6894$) and Fine-Grained ($0.6145$). Temporal shift caused sharp Macro-F1 drop across all models. |
| **Stage 5** | [`STAGE_5_ERROR_ANALYSIS.md`](STAGE_5_ERROR_ANALYSIS.md) | Fine-grained confusion matrix inspection, worst-10 attack classes, false negatives against Benign, and class weighting ablations. | OS Reconnaissance exhibited 21.06% misclassification to Benign. Web attacks suffered from flow statistical payload blindness. Unweighted models collapsed minority classes. |
| **Stage 6** | [`STAGE_6_RESEARCH_SYNTHESIS.md`](STAGE_6_RESEARCH_SYNTHESIS.md) | Comprehensive research findings synthesis, research contributions, limitations, and calibrated scientific claims. | Clarified published literature accuracy context, qualified Strategy 2 gains as marginal ($\Delta=+0.0013$), and differentiated demonstrated findings from hypotheses. |
| **Stage 7** | [`STAGE_7_LITERATURE_REVIEW.md`](STAGE_7_LITERATURE_REVIEW.md) | Systematic review of 14 peer-reviewed research papers and capstone technical documentation on CICIoT2023. | Identified critical methodological gaps in prior literature: lack of duplicate auditing, absence of collision handling, and reliance on unpartitioned random splits. |

---

## 2. Supporting System Audits

- **[`PROJECT_ORIENTATION_REPORT.md`](PROJECT_ORIENTATION_REPORT.md)**: Initial codebase orientation audit examining the legacy Generation 1 pipeline, schema consistency (39 features vs. legacy 46), and computational requirements.
- **[`PROJECT_STRUCTURE_AUDIT.md`](PROJECT_STRUCTURE_AUDIT.md)**: Complete read-only audit of all 357 project files across 34 directories, cataloging exact duplicates, directory interdependencies, and architectural evolution.

---

## 3. Data & Artifact Cross-References

- **Active Model Checkpoints & Preprocessors**: `models/active/` (all 21 Stage 3 `.joblib` and `.pt` models, plus `preprocessor_strategy1.joblib` and `preprocessor_strategy2.joblib`).
- **Evaluation Tables & Centralized Ledger**: `results_v2/tables/centralized_experiment_ledger.csv` and `results_v2/tables/stage4_test_results.csv`.
- **Per-Class Metrics & Loss Curves**: `results_v2/metrics/per_class/` and `results_v2/metrics/stage4/`.
- **Partition Assignment Maps**: `data/partitions/` and `data/partitions/ablation_strategy2/`.
