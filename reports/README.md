# Master Research Reports & Methodological Audits (Stages 1–8D)

This directory houses the complete chronological archive of research reports, forensic audits, verification documents, and manuscript blueprints produced across the capstone project investigating Machine Learning and Deep Learning baselines for IoT Network Intrusion Detection on the **CICIoT2023** benchmark.

---

## 1. Master Research Trajectory (Stages 1–8D)

| Stage | Report Document | Scope & Methodology | Primary Finding / Calibrated Output |
|:---:|---|---|---|
| **Stage 1** | [`STAGE_1_AUDIT_REPORT.md`](STAGE_1_AUDIT_REPORT.md) | Forensic dataset audit of 529,562 flows from `sample_stratified.csv`. Quantified exact duplicates, identical feature vectors, and label collisions. | Identified 8,740 unique feature hashes mapped to conflicting labels, with 48.2% involving `BenignTraffic`. Demonstrated that naive random splitting causes identical-feature data leakage. |
| **Stage 2** | [`STAGE_2_INVESTIGATION_REPORT.md`](STAGE_2_INVESTIGATION_REPORT.md) | Architectural design of atomic feature-hash grouping, collision purging (Strategy 2), and disjoint temporal cross-capture partition (`Merged26`–`Merged40`). | Proved 0% feature-vector leakage across splits. Constructed clean temporal cross-capture test set (N = 57,280) purged of all training collision hashes. |
| **Stage 3** | [`STAGE_3_COMPLETE_REPORT.md`](STAGE_3_COMPLETE_REPORT.md) | Comprehensive consolidation of Stage 3: baseline model training across 5 phases, infrastructure canary, preprocessor isolation, decomposed validation routing, and resource profiling. | Benchmarked 21 models. LightGBM led Category (0.6865) and Fine-Grained (0.6142); Random Forest led Binary (0.7733). Decomposed routing proved Strategy 2 clean-set gain is marginal (Δ = +0.0022). |
| **Stage 4** | [`STAGE_4_EVALUATION_REPORT.md`](STAGE_4_EVALUATION_REPORT.md) | Read-only dual-test evaluation of all 21 Stage 3 model checkpoints on Internal Test (N = 75,105) and same-testbed Temporal Cross-Capture (N = 57,280). | Random Forest led binary internal test Macro-F1 (0.7690); LightGBM led Category (0.6894) and Fine-Grained (0.6145). Temporal shift caused marked Macro-F1 drop across all binary models. |
| **Stage 5** | [`STAGE_5_ERROR_ANALYSIS.md`](STAGE_5_ERROR_ANALYSIS.md) | Fine-grained confusion matrix inspection, worst-10 attack classes, false negatives against Benign, and class weighting ablations. | OS Reconnaissance exhibited 21.06% misclassification to Benign. Web attacks suffered from flow statistical payload blindness. Unweighted models collapsed minority classes. |
| **Stage 6** | [`STAGE_6_RESEARCH_SYNTHESIS.md`](STAGE_6_RESEARCH_SYNTHESIS.md) | Comprehensive research findings synthesis, research contributions, limitations, and calibrated scientific claims. | Clarified published literature accuracy context, qualified Strategy 2 gains as marginal (Δ = +0.0013), and differentiated demonstrated findings from hypotheses. |
| **Stage 7** | [`STAGE_7_LITERATURE_REVIEW.md`](STAGE_7_LITERATURE_REVIEW.md) | Systematic review of 14 peer-reviewed research papers and capstone technical documentation on CICIoT2023. | Identified critical methodological gaps in prior literature: lack of duplicate auditing, absence of collision handling, and reliance on unpartitioned random splits. |
| **Stage 8A** | [`STAGE_8A_RESEARCH_CONSOLIDATION.md`](STAGE_8A_RESEARCH_CONSOLIDATION.md) | Comprehensive research consolidation across Stages 1–7. Formulates central RQ, defines 7 defensible contributions, calibrates claim boundaries, and outlines paper identity. | Established unified research problem, validated research contributions against verified metrics, defined Category A/B/C claim boundaries, and mapped publication roadmap. |
| **Stage 8A.1**| [`STAGE_8A1_CONSISTENCY_AUDIT.md`](STAGE_8A1_CONSISTENCY_AUDIT.md) | Independent scientific consistency audit of the Stage 8A consolidation. Audited all numerical claims, uncovered 18 discrepancies, resolved cross-capture metric paradox, and calibrated contributions. | Disproved 78.9% redundancy (true: 11.29%), proved simulated collision leakage is 81.39% (not 100%), corrected class-swapped ablation recalls, and aligned telemetry with raw CSV ledgers. |
| **Stage 8A.2**| [`STAGE_8A2_AUDIT_VERIFICATION.md`](STAGE_8A2_AUDIT_VERIFICATION.md) | Focused verification and mathematical reconciliation of Stage 8A.1 findings. Deterministically verified duplicate arithmetic, leakage, cross-capture shift, and class weighting. | Reconciled 18,416 vs. 28,608 collision rows (post- vs. pre-deduplication), verified 81.39% simulated test collision leakage, proved cross-capture Macro-F1 rise is driven by prevalence shift, and certified ground-truth constants. |
| **Stage 8B** | [`STAGE_8B_MANUSCRIPT_PLAN.md`](STAGE_8B_MANUSCRIPT_PLAN.md) | Authoritative manuscript blueprint and publication-oriented drafting plan. Establishes final title, Central RQ, 4 consolidated contributions, 9-section plan, 14-paper literature matrix, 3-tier claim ledger, and figure/table specifications. | Direct execution bridge for Stage 8C manuscript drafting. Incorporates all Stage 8A.2 certified numerical ground truths with zero unresolved contradictions. |
| **Stage 8C** | [`STAGE_8C_DRAFTING_NOTES.md`](STAGE_8C_DRAFTING_NOTES.md) | Manuscript drafting execution notes, deliverable inventory, and verification checklist. | Completed full drafts of `paper/main.tex` and `paper/paper.typ`, generated publication figures 1–6, and updated `references.bib`. |
| **Stage 8D** | [`STAGE_8D_FINAL_REVIEW.md`](STAGE_8D_FINAL_REVIEW.md) | Final manuscript audit, computational profiling reconciliation, and packaging report. | Reconciled all P50/P95 latencies and throughputs against raw ledgers, eliminated promotional phrasing, compiled 11-page publication PDF (`research_paper.pdf`). |
| **Sync** | [`STAGE_8D_REPOSITORY_SYNCHRONIZATION.md`](STAGE_8D_REPOSITORY_SYNCHRONIZATION.md) | Repository-wide synchronization report aligning documentation, scripts, and findings. | Verified complete repository consistency, cross-references, and clean audit state across all directories. |
| **Stage 9** | [`STAGE_9_FINAL_PUBLICATION_VERIFICATION.md`](STAGE_9_FINAL_PUBLICATION_VERIFICATION.md) | Final publication verification, PDF audit, and approval report. | Independent audit of publication PDF deliverable, visual inspection across all 11 pages, resolution of 10 targeted manuscript issues, citation parity verification, and final submission sign-off. |

---

## 2. Supporting Architecture & Pre-Execution Audits

- **[`PROJECT_ORIENTATION_REPORT.md`](PROJECT_ORIENTATION_REPORT.md)**: Initial codebase orientation audit examining the legacy Generation 1 pipeline, schema consistency (39 features vs. legacy 46), and computational requirements.
- **[`PROJECT_STRUCTURE_AUDIT.md`](PROJECT_STRUCTURE_AUDIT.md)**: Comprehensive structural audit of all project files, cataloging exact duplicates, directory interdependencies, and architectural evolution.

---

## 3. Artifact & Ledger Cross-References

- **Active Model Checkpoints & Preprocessors**: `models/active/` (all 21 Stage 3 `.joblib` and `.pt` models, plus `preprocessor_strategy1.joblib` and `preprocessor_strategy2.joblib`).
- **Evaluation Tables & Centralized Ledger**: `results_v2/tables/centralized_experiment_ledger.csv` and `results_v2/tables/stage4_test_results.csv`.
- **Per-Class Metrics & Loss Logs**: `results_v2/metrics/per_class/` and `results_v2/metrics/stage4/`.
- **Partition Assignment Maps**: `data/partitions/` and `data/partitions/ablation_strategy2/`.
