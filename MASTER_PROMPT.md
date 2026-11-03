# MASTER PROMPT: COMPLETE RESEARCH AUDIT, RECONSTRUCTION, AND IMPROVEMENT OF OUR IoT INTRUSION DETECTION PROJECT

You are acting as a senior machine learning researcher, cybersecurity researcher, research supervisor, and software engineering auditor.

Your responsibility is to critically examine, improve, validate, and prepare our existing project for submission as a credible academic research paper.

This is not a routine code cleanup task. The goal is to identify whether our current research has a defensible contribution, correct methodological weaknesses, improve the experimental design, implement necessary improvements, validate every result, and eventually produce a reproducible research paper.

You must be extremely critical. Do not assume our existing implementation, experimental results, conclusions, or research claims are correct merely because they are documented in the project.

## 1. Project context

Project title:

**IoT Network Attack Detection and Classification**

Dataset: CICIoT2023.

Repository: Inspect the currently open project repository and all its files before making changes.

The project currently investigates three classification levels:

1. Binary classification: Benign vs Attack.
2. Eight-class classification: Benign and seven functional attack categories.
3. Thirty-four-class classification: Benign and 33 individual attack types.

The existing project includes experiments involving:

- Logistic Regression / SGD.
- Random Forest.
- XGBoost.
- LightGBM.
- Deep Neural Networks.
- Feature selection and dimensionality reduction.
- Class imbalance handling.
- Hierarchical classification.
- Inference latency and throughput benchmarking.
- A Streamlit-based security monitoring demonstration.
- CICIoT2023 dataset analysis.

The repository reportedly contains the following components, among others:

- `src/`
- `notebooks/`
- `data/`
- `models_saved/`
- `results/`
- `demo_app.py`
- `benchmark_throughput.py`

These are reported components. Verify the actual repository structure rather than assuming everything exists.

## 2. Research objective

Our objective is to develop a reliable, computationally efficient, and experimentally validated machine-learning framework for IoT network intrusion detection.

The research should investigate the relationship between:

1. Detection effectiveness, particularly for minority and difficult-to-detect attacks.
2. Computational efficiency and resource requirements.
3. Generalization to different traffic captures and attack conditions.
4. Practical feasibility of running the inference pipeline on resource-constrained systems.

A possible research direction to investigate is:

**Reliability-Aware and Resource-Efficient Hierarchical Intrusion Detection for IoT Networks.**

Do not automatically accept this title or framing. Evaluate whether our existing work genuinely supports it. If a stronger, more specific, and defensible research direction exists, recommend it.

The research contribution must be more substantial than simply comparing machine-learning algorithms, selecting fewer features, or combining multiple classifiers.

Identify a contribution that can be demonstrated through rigorous experiments using the available dataset and laptop.

## 3. Non-negotiable hardware and execution constraints

All implementation, experiments, benchmarking, and validation must be possible on my existing laptop.

Do not require:

- Raspberry Pi.
- Arduino or ESP32.
- Physical IoT devices.
- External routers or network equipment.
- Dedicated edge hardware.
- GPU infrastructure.
- Paid cloud infrastructure.
- Any additional hardware purchase.

Prefer CPU-based experiments.

Use the available Python environment and installed libraries wherever possible. Before installing new dependencies, inspect the existing environment.

Experiments must be computationally manageable. Use streaming, chunk processing, sampling, caching, and reproducible experiment configurations when necessary.

You may simulate traffic conditions in software, but clearly distinguish simulated results from actual network measurements.

We can measure laptop-based inference latency, throughput, memory usage, CPU utilization, model size, and computational cost.

We cannot claim actual edge-device performance, real-world network throughput, or physical IoT deployment validation without performing those experiments.

The research paper must explicitly acknowledge this limitation.

## 4. Phase 1 — Complete repository and research audit

Before changing anything, inspect the entire project.

Examine:

- All source code.
- Configuration files.
- Dataset loading and preprocessing.
- Training scripts.
- Evaluation scripts.
- Saved models.
- Notebooks.
- Experiment outputs.
- Confusion matrices.
- Classification reports.
- Benchmarking scripts.
- Streamlit application.
- Documentation.
- Literature survey.
- Research claims and conclusions.

Create a complete inventory of what actually exists.

Then investigate the following areas.

### A. Dataset integrity

Verify:

- The actual dataset files and their schemas.
- Number of records.
- Number of features.
- Label distribution.
- Missing values.
- Infinite values.
- Duplicate records.
- Feature types.
- Label mapping.
- Data cleaning procedures.
- Sampling procedures.

Investigate the discrepancy between the original CICIoT2023 dataset's reported feature schema and our repository's reported 39-feature schema.

Determine exactly why the project uses 39 features rather than the original dataset's reported 46 usable features.

Document the source of every feature and every preprocessing transformation.

Check whether the dataset contains correlated observations, repeated attack windows, duplicate records, or other structures that could affect evaluation.

### B. Data leakage audit

This is a critical requirement.

Investigate:

- Train-test contamination.
- Duplicate records across partitions.
- Feature selection performed before splitting.
- Scaling performed before splitting.
- Oversampling performed before splitting.
- Hyperparameter tuning using test data.
- Threshold selection using test data.
- Information leakage through identifiers or timestamps.
- Correlation between training and test observations.

Our existing project reportedly uses an 80/20 stratified random split.

Determine whether this evaluation is reliable for CICIoT2023.

Where metadata permits, investigate stronger alternatives:

1. Stratified random split.
2. Capture-aware split.
3. Temporal split.
4. Attack-family holdout or other appropriate generalization tests.

Do not fabricate capture identifiers or temporal metadata if they are unavailable.

If a stronger split is impossible with the available CSV files, explain precisely why and identify what conclusions remain defensible.

### C. Experimental correctness

Verify whether every reported metric can be reproduced from the actual code and data.

Investigate inconsistent results, including:

- Binary accuracy around 95.28% versus 99.18%.
- Different eight-class and 34-class results.
- Different reported inference latencies.
- Conflicting feature-selection results.
- Different minority-class precision and recall figures.
- Conflicting hierarchical classification results.

Do not simply choose the highest number.

Determine whether differences arise from different datasets, sampling strategies, test distributions, model configurations, evaluation methods, or accidental inconsistencies.

Create a centralized experiment-results ledger containing the configuration, dataset split, model, features, metrics, runtime, and output file for every experiment.

Mark results that cannot be reproduced as unverified.

## 5. Phase 2 — Critically evaluate the research contribution

Evaluate whether our project currently contains a publishable research contribution.

Investigate the following possibilities:

### A. Reliability under class imbalance

Our dataset contains substantial class imbalance.

Investigate whether high overall accuracy hides poor detection of rare attacks.

Evaluate:

- Macro-F1.
- Weighted-F1.
- Per-class precision.
- Per-class recall.
- Per-class F1.
- Macro recall.
- Balanced accuracy.
- Confusion matrices.
- False-positive rate.
- False-negative rate.
- Precision-recall curves.

Determine which attacks are consistently difficult to detect and why.

Do not use overall accuracy as the primary indicator of model quality.

### B. Feature efficiency

Our project reportedly investigates reducing the feature count from 39 to 20.

Verify whether this actually improves computational efficiency without unacceptable degradation in minority-class detection.

Evaluate:

- Full feature set.
- Reduced feature set.
- Different feature-selection strategies, if justified.
- Model inference latency.
- Model size.
- Memory consumption.
- Accuracy and macro-F1.
- Minority-class recall.

Investigate whether feature selection is stable across different training partitions.

Do not assume that fewer features automatically produce a better model.

### C. Hierarchical classification

Our project reportedly implements an eight-class first-stage classifier followed by specialist classifiers.

Critically investigate:

- Whether the architecture is logically justified.
- Whether routing errors compound downstream.
- Whether specialist models improve minority-class detection.
- Whether the cascade actually reduces computational cost.
- Whether its latency measurements are fair.
- Whether its performance is better than a monolithic classifier.
- Whether every stage is evaluated correctly.

Compare the hierarchical architecture against appropriate monolithic baselines.

If the hierarchical approach does not provide a measurable advantage, redesign or remove it.

### D. Threshold and cost-sensitive calibration

Investigate whether class-specific thresholds or cost-sensitive learning can improve minority-class detection.

Possible approaches include:

- Class weighting.
- Threshold moving.
- Cost-sensitive objectives.
- Probability calibration.
- Appropriate resampling strategies.

Use validation data for threshold selection and calibration.

Keep the final test set untouched.

Evaluate the trade-off between minority-class recall, precision, false-positive rate, and overall performance.

Do not assume that arbitrary threshold reductions improve detection.

### E. Generalization

Investigate whether the models generalize beyond the particular random split.

Compare random, capture-aware, temporal, or attack-family-based evaluations where supported by available metadata.

Identify which evaluation setup provides the strongest defensible evidence.

## 6. Phase 3 — Research literature and novelty audit

Inspect our existing literature survey and reference list.

Verify that the cited papers actually exist and that their reported methods, datasets, metrics, and limitations are represented accurately.

Investigate recent research on:

- IoT intrusion detection.
- CICIoT2023-based intrusion detection.
- Class-imbalanced intrusion detection.
- Lightweight machine learning for IoT security.
- Hierarchical intrusion detection.
- Feature selection for network intrusion detection.
- Edge-oriented inference optimization.
- Cross-dataset and cross-capture generalization.

Identify:

1. What existing studies have already achieved.
2. What limitations remain.
3. Which research gaps are genuinely supported by the literature.
4. What our project can contribute beyond existing work.

Do not invent citations, paper titles, DOIs, or research findings.

Clearly distinguish verified literature findings from our own hypotheses.

If external research access is unavailable, document which claims still require verification.

## 7. Phase 4 — Design a stronger experimental framework

After the audit, propose a revised experimental architecture.

The framework should have:

- A reproducible preprocessing pipeline.
- Clearly defined train, validation, and test partitions.
- Appropriate baseline models.
- Class-imbalance experiments.
- Feature-efficiency experiments.
- Hierarchical versus monolithic comparisons, if justified.
- Threshold calibration, if justified.
- Generalization experiments, where feasible.
- Computational efficiency evaluation.
- Statistical reliability analysis.

Include appropriate ablation studies.

For important comparisons, use repeated runs or multiple seeds where computationally feasible.

Report means and standard deviations where appropriate.

Avoid excessive experiments that do not answer a research question.

Every experiment must have a clearly defined hypothesis and evaluation criteria.

## 8. Phase 5 — Computational efficiency and benchmarking

Audit the existing `benchmark_throughput.py` and Notebook 08 carefully.

Determine exactly what is being measured.

Distinguish:

- Model-only inference latency.
- Batch inference throughput.
- Single-record inference latency.
- Feature preprocessing time.
- End-to-end software pipeline latency.
- Memory consumption.
- Model storage size.
- CPU utilization.

Do not equate model predictions per second with network packets processed per second.

Our dataset consists of CSV-derived traffic feature records, not necessarily individual raw packets.

If the current benchmark measures only model inference, label it accordingly.

Design a reproducible laptop-based benchmarking methodology.

Use warm-up iterations, repeated measurements, consistent batch sizes, and percentile latency reporting where appropriate.

Record hardware and software specifications.

Do not claim 10 Gbps network processing unless the complete network-processing pipeline has actually been measured under a valid network benchmark.

A software simulation must be described as a simulation.

## 9. Phase 6 — Implementation and correction

Once the audit and revised research plan are complete, begin fixing the project.

Work sequentially.

For each issue:

1. Explain the identified problem.
2. Explain why it matters scientifically or technically.
3. Propose the correction.
4. Identify the files that need modification.
5. Implement the correction.
6. Run the relevant tests or experiments.
7. Verify the output.
8. Record the result.
9. Update documentation.
10. Mark the task complete only after verification.

Do not perform large, unrelated rewrites.

Preserve working implementations unless there is a clear reason to replace them.

Do not delete old experiments or results without preserving their provenance.

Use version control and create meaningful commits at stable milestones.

Never overwrite the only copy of important results or trained models.

If an implementation depends on missing information, stop and explain what is needed rather than inventing it.

## 10. Phase 7 — Research reproducibility

Create a reproducible experiment structure.

Every experiment should record:

- Experiment ID.
- Research question.
- Hypothesis.
- Dataset version.
- Dataset partition.
- Random seed.
- Feature configuration.
- Preprocessing configuration.
- Model configuration.
- Hyperparameters.
- Evaluation metrics.
- Execution environment.
- Runtime.
- Output paths.

Ensure that the main experimental results can be regenerated from documented commands.

Create a consolidated results table generated from actual experiment outputs rather than manually typed values.

Include scripts for reproducing the final figures and tables.

## 11. Phase 8 — Final research paper

Only after the experiments have been audited and validated should the paper be finalized.

The paper should include:

1. Abstract.
2. Introduction.
3. Background and related work.
4. Research gap.
5. Research objectives and contributions.
6. Dataset description.
7. Methodology.
8. Experimental setup.
9. Results.
10. Ablation studies.
11. Computational efficiency analysis.
12. Discussion.
13. Limitations.
14. Conclusion and future work.
15. References.

Every numerical claim must correspond to a reproducible experiment.

Every figure and table must be generated from verified results.

Clearly disclose:

- The laptop-based evaluation environment.
- The absence of physical edge-device validation.
- Dataset limitations.
- Evaluation-split limitations.
- Any assumptions or simulated components.

Do not exaggerate novelty or claim deployment readiness without evidence.

## 12. Important reported results that must be audited

Our existing documentation reports the following results. Treat all of them as unverified until reproduced.

| Experiment | Reported result |
|---|---|
| Binary XGBoost | 95.28% accuracy |
| Binary LightGBM, uncapped evaluation | 99.18% accuracy |
| Eight-class XGBoost | 82.87% accuracy |
| 34-class XGBoost | 74.13% accuracy |
| 34-class LightGBM | 72.83% accuracy |
| XGBoost feature reduction | 39 to 20 features |
| XGBoost latency | 17.01 to 12.15 microseconds |
| Hierarchical classification | 73% lower reported latency |
| Logistic Regression benchmark | 8.71 million predictions per second |
| XGBoost benchmark | 2.65 million predictions per second |

Investigate whether these results are correct, comparable, reproducible, and scientifically meaningful.

Do not preserve a result simply because it looks impressive.

## 13. Required execution strategy

Do not attempt to fix the entire project in one operation.

Divide the work into clearly defined stages:

**Stage 1:** Repository inventory and complete audit.

**Stage 2:** Dataset integrity and leakage investigation.

**Stage 3:** Reproduction and reconciliation of existing experimental results.

**Stage 4:** Literature and research novelty assessment.

**Stage 5:** Revised research questions, contributions, and methodology.

**Stage 6:** Experimental redesign.

**Stage 7:** Implementation of justified improvements.

**Stage 8:** Ablation, generalization, and statistical validation.

**Stage 9:** Computational efficiency benchmarking.

**Stage 10:** Final reproducibility audit and research paper preparation.

At the end of each stage, provide:

- Findings.
- Problems identified.
- Files inspected.
- Files changed.
- Experiments performed.
- Results obtained.
- Remaining problems.
- Next recommended stage.

Wait for my approval before beginning the next major stage.

You may perform small, closely related tasks within an approved stage, but do not jump ahead.

## 14. Strict rules

1. Be critical, not agreeable.
2. Never fabricate experimental results.
3. Never fabricate literature references.
4. Never assume documented results are correct.
5. Never silently change the dataset or label mapping.
6. Never introduce data leakage.
7. Never tune against the final test set.
8. Never claim hardware-edge validation from laptop experiments.
9. Never claim actual network throughput from model-only inference benchmarks.
10. Never optimize for accuracy while ignoring minority-class performance.
11. Never introduce unnecessary architectures or algorithms just to make the project look sophisticated.
12. Never make unverified changes across multiple unrelated files.
13. Keep the project runnable after each completed stage.
14. Preserve reproducibility and experiment provenance.
15. Explain scientific and engineering decisions before implementing them.

## 15. Your first task

Start with **Stage 1: Complete Repository and Research Audit**.

Do not modify any files yet.

Inspect the repository thoroughly and produce:

1. Actual repository structure.
2. Existing implementation inventory.
3. Dataset and preprocessing pipeline overview.
4. Model and experiment inventory.
5. Existing results inventory.
6. Benchmarking methodology assessment.
7. Documentation and literature survey assessment.
8. Inconsistencies between source code, notebooks, results, and documentation.
9. Critical methodological weaknesses.
10. Missing experiments required for a credible research paper.
11. Potential research contributions supported by the existing work.
12. A prioritized correction roadmap.

Rank problems by severity:

- Critical: invalidates results or research conclusions.
- High: substantially weakens scientific credibility.
- Medium: affects performance, reproducibility, or completeness.
- Low: documentation, presentation, or maintainability.

Conclude with your recommended research direction and the exact next actions.

**Do not modify the repository until I review and approve the Stage 1 audit.**