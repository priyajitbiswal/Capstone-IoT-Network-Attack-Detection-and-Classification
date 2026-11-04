# Stage 6: Research Findings and Contribution Synthesis

**Project Title**: Empirical Evaluation and Methodological Auditing of Machine Learning Baselines for IoT Network Intrusion Detection on the CICIoT2023 Benchmark  
**Artifact Directory**: `results_v2/` | **Ledger**: [`results_v2/tables/centralized_experiment_ledger.csv`](tables/centralized_experiment_ledger.csv)  
**Final Test Table**: [`results_v2/tables/stage4_test_results.csv`](tables/stage4_test_results.csv)  

---

## 1. Research Problem and Scope

### 1.1 Problem Definition
The Internet of Things (IoT) encompasses heterogeneous, resource-constrained devices increasingly targeted by automated cyberattacks. The **CICIoT2023** dataset was introduced as a modern benchmark comprising 33 attack profiles and benign traffic recorded across 105 physical devices, structured into 46 network flow features.

However, standard benchmark pipelines frequently exhibit methodological vulnerabilities that can affect empirical evaluation and reproducibility:
1. **Pervasive Feature Vector Redundancy and Potential Data Leakage**: Standard random train-test splits can place identical feature vectors into both training and evaluation partitions when substantial feature redundancy exists. While our experiments do not evaluate individual published papers to determine whether specific results above 99% were caused by leakage, data leakage arising from duplicate feature vectors is a known mechanism that can artificially inflate evaluation metrics.
2. **Conflicting Label Collisions**: Thousands of network flows share identical feature vectors while carrying contradictory ground-truth labels (e.g., identical flow features assigned to both `BENIGN` and `DOS-TCP_FLOOD`). These conflicting vectors introduce label noise into training and bound classifier performance on ambiguous instances.
3. **Preprocessing Contamination**: Computing transformation statistics (such as mean, variance, and extreme-value caps) across an entire dataset prior to partitioning leaks distribution information from evaluation sets into training routines.
4. **Accuracy Masking under Extreme Class Imbalance**: Volumetric DDoS/DoS attacks account for over $54\%$ of total traffic in the benchmark sample, whereas minority web and exploit intrusions represent under $2.3\%$. Optimizing solely for aggregate accuracy can mask substantial recall degradation on minority attack classes.
5. **Absence of Temporal Cross-Session Validation**: Evaluating models only on holdouts sampled from the same capture sessions leaves their stability across distinct recording sessions and temporal variations unmeasured.

### 1.2 Scope of this Investigation
This project conducts an empirical evaluation and methodological audit of machine learning baselines on the CICIoT2023 benchmark. The scope encompasses:
- An audit of data integrity, duplicate instances, and conflicting label collisions across 529,562 flow records (Stage 1 Report).
- Construction of an isolated, feature-hash partitioned experimental testbed (Stage 2 Report).
- Training, validation, and computational profiling of **21 model configurations** across 5 phases spanning three attack granularities (Binary, 8-Class Category, and 34-Class Fine-Grained) and two partitioning strategies (Stage 3 Ledger).
- Evaluation on both an **Internal In-Distribution Test Set ($N = 75,105$)** and a **Same-Testbed Temporal Cross-Capture Clean Test Set ($N = 57,280$)** (Stage 4 Report).
- A fine-grained confusion, false-negative, and ablation error analysis (Stage 5 Reports).

---

## 2. Research Questions

- **RQ1 (Partitioning & Integrity)**: *To what extent do duplicate feature vectors and label collisions occur within the CICIoT2023 sample, and can feature-hash partitioning prevent inter-partition feature overlap while maintaining representative class distributions?*
- **RQ2 (Architectural Performance by Granularity)**: *How do linear models (SGD), shallow neural networks (PyTorch MLP), and tree ensembles (Random Forest, LightGBM, XGBoost) compare in Macro-F1, Accuracy, Matthews Correlation Coefficient (MCC), and inference efficiency across Binary, 8-Class Category, and 34-Class Fine-Grained tasks?*
- **RQ3 (Label Ambiguity & Collision Purging)**: *What is the empirical effect of training with contradictory collision vectors (Strategy 1: Primary) versus purging ambiguous collision clusters (Strategy 2: Collision-Purged) on in-distribution test performance and same-testbed temporal cross-capture stability?*
- **RQ4 (Class Imbalance & Objective Weighting)**: *How does inverse-frequency class weighting influence minority attack recall, overall accuracy, and false-negative rates to benign traffic compared to unweighted optimization?*
- **RQ5 (Temporal Cross-Capture Stability)**: *How do models trained on early capture sessions perform on subsequent capture sessions (`Merged26.csv`–`Merged40.csv`) from the same IoT testbed, and which functional threat tiers exhibit the greatest performance variation across sessions?*

---

## 3. Synthesis of Experimental Evidence (Stages 1–5)

### 3.1 Dataset Integrity and Partitioning (Stages 1 & 2)
The audit of the 529,562 raw flow records from `data/sample_stratified.csv` identified three distinct categories of redundancy:
- **Exact Duplicate Rows**: $28,464$ rows ($5.375\%$) were full-record exact duplicates (identical features and identical label).
- **Redundant Feature Vectors with Identical Labels**: $9,676$ unique feature vectors spanning $31,172$ rows ($5.886\%$) had identical feature values and agreed on label assignments.
- **Conflicting Label Collisions**: **$8,740$ unique feature hashes** across **$28,608$ rows** ($5.402\%$ of the 529,562-row sample) mapped to contradictory ground-truth labels.
- **Partitioning Design**: In Stage 2 (`src_v2/data_partitioning.py`), an atomic feature-hash grouping protocol was implemented to ensure that all occurrences of an identical feature hash reside strictly within a single partition:
  $$\text{Overlap}(\mathcal{D}_{\text{train}}, \mathcal{D}_{\text{val}}) = 0, \quad \text{Overlap}(\mathcal{D}_{\text{train}}, \mathcal{D}_{\text{test}}) = 0, \quad \text{Overlap}(\mathcal{D}_{\text{val}}, \mathcal{D}_{\text{test}}) = 0$$
- **Two Partition Strategies**:
  - **Strategy 1 (Primary / Collision-Contaminated)**: Retains all $501,098$ unique rows ($N_{\text{train}} = 350,821$; $N_{\text{val}} = 75,172$; $N_{\text{test}} = 75,105$).
  - **Strategy 2 (Ablation / Collision-Purged)**: Purges all 8,740 ambiguous collision hashes ($N_{\text{train}} = 337,838$; $N_{\text{val}} = 72,390$; $N_{\text{test}} = 72,454$).
  - **Common Clean Subset**: The 72,390 clean validation flows in Strategy 1 correspond exactly to the Strategy 2 validation population ($\mathcal{D}_{\text{val, clean}} \equiv \mathcal{D}_{\text{val, S2}}$), providing a shared subpopulation for cross-strategy comparison.

### 3.2 Preprocessing Provenance and Scaling Verification
- In Stage 3, feature preprocessing was handled by `LeakFreePreprocessor` (`src_v2/preprocessing.py`).
- Preprocessor inspection in Stage 4 confirmed that `scale_features=True` was passed during Stage 3 data ingestion. Consequently, **all 21 models—including tree-based ensembles—were trained on standardized $z$-score features** ($\mu = 0, \sigma = 1$).
- Preprocessing parameters (`column_finite_max` and `StandardScaler` moments $\mu, \sigma^2$) were exported to:
  - `models_saved_v2/preprocessor_strategy1.joblib` ($N_{\text{train}} = 350,821$)
  - `models_saved_v2/preprocessor_strategy2.joblib` ($N_{\text{train}} = 337,838$)
- Parity verification confirmed that transforming validation flows with these exported preprocessors reproduces all 21 Stage 3 validation metrics with exact numerical agreement ($\Delta = 0.00000$). When tree models were evaluated on unscaled raw inputs, accuracy dropped significantly (e.g., `fine_grained_lightgbm` accuracy dropped from $72.46\%$ to $7.31\%$), confirming that tree split thresholds were learned in $z$-score space.

### 3.3 Final Test Evaluation Across Partitions (Stage 4)
All 21 model checkpoints were evaluated across two test partitions without refitting or post-hoc threshold tuning:

#### Consolidated Performance Table Across Partitions
*Metrics from `results_v2/tables/stage4_test_results.csv`:*

| Experiment ID | Model Architecture | Strategy | Task Scope | Val Macro-F1 | Internal Test Acc | Internal Test F1 | Internal Test MCC | Cross-Capture Acc | Cross-Capture F1 | Cross-Capture MCC | $\Delta_{\text{gen}}$ F1 | $\Delta_{\text{shift}}$ F1 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `binary_logistic_regression` | SGD (Linear) | Strategy 1 | Binary | 0.6803 | 81.73% | 0.6761 | 0.4849 | 73.88% | 0.6137 | 0.4104 | -0.0042 | -0.0624 |
| `binary_random_forest` | Random Forest | Strategy 1 | Binary | 0.7733 | 89.61% | **0.7690** | 0.6057 | 84.30% | **0.7108** | 0.5245 | -0.0043 | -0.0582 |
| `binary_lightgbm` | LightGBM | Strategy 1 | Binary | 0.7665 | 89.15% | 0.7637 | 0.6007 | 83.48% | 0.7034 | 0.5189 | -0.0028 | -0.0603 |
| `binary_xgboost` | XGBoost | Strategy 1 | Binary | 0.7501 | 87.79% | 0.7461 | 0.5785 | 81.03% | 0.6789 | 0.4898 | -0.0040 | -0.0672 |
| `binary_pytorch_dnn` | PyTorch MLP (5 ep) | Strategy 1 | Binary | 0.7143 | 85.11% | 0.7125 | 0.5315 | 78.33% | 0.6525 | 0.4560 | -0.0018 | -0.0600 |
| `ablation_s2_binary_lightgbm` | LightGBM | Strategy 2 | Binary | 0.7686 | 89.39% | 0.7668 | 0.6045 | 83.75% | 0.7062 | 0.5219 | -0.0018 | -0.0607 |
| `category_logistic_regression` | SGD (Linear) | Strategy 1 | 8-Class | 0.6006 | 74.04% | 0.6036 | 0.6723 | 68.53% | 0.6084 | 0.6204 | +0.0031 | +0.0047 |
| `category_random_forest` | Random Forest | Strategy 1 | 8-Class | 0.6802 | 78.22% | 0.6805 | 0.7399 | 76.25% | 0.7276 | 0.7211 | +0.0004 | +0.0470 |
| `category_lightgbm` | LightGBM | Strategy 1 | 8-Class | 0.6865 | 78.93% | **0.6894** | 0.7468 | 77.17% | **0.7399** | 0.7315 | +0.0028 | +0.0505 |
| `category_xgboost` | XGBoost | Strategy 1 | 8-Class | 0.6615 | 76.08% | 0.6619 | 0.7180 | 74.02% | 0.7120 | 0.6992 | +0.0004 | +0.0501 |
| `category_pytorch_dnn` | PyTorch MLP (5 ep) | Strategy 1 | 8-Class | 0.6104 | 73.41% | 0.6130 | 0.6834 | 71.09% | 0.6578 | 0.6619 | +0.0026 | +0.0448 |
| `ablation_s2_category_lightgbm` | LightGBM | Strategy 2 | 8-Class | 0.6925 | 79.12% | **0.6903** | 0.7482 | 77.30% | **0.7411** | 0.7326 | -0.0022 | +0.0508 |
| `fine_grained_logistic_regression` | SGD (Linear) | Strategy 1 | 34-Class | 0.5260 | 64.29% | 0.5272 | 0.6308 | 59.49% | 0.5501 | 0.5817 | +0.0012 | +0.0228 |
| `fine_grained_random_forest` | Random Forest | Strategy 1 | 34-Class | 0.5956 | 70.58% | 0.5966 | 0.6972 | 67.52% | 0.6297 | 0.6653 | +0.0010 | +0.0331 |
| `fine_grained_lightgbm` | LightGBM | Strategy 1 | 34-Class | 0.6142 | 72.65% | **0.6145** | 0.7170 | 69.64% | 0.6498 | 0.6861 | +0.0003 | +0.0353 |
| `fine_grained_xgboost` | XGBoost | Strategy 1 | 34-Class | 0.5984 | 71.04% | 0.6001 | 0.7012 | 67.87% | 0.6341 | 0.6685 | +0.0016 | +0.0340 |
| `fine_grained_pytorch_dnn` | PyTorch MLP (5 ep) | Strategy 1 | 34-Class | 0.5211 | 62.42% | 0.5216 | 0.6137 | 59.52% | 0.5470 | 0.5842 | +0.0005 | +0.0254 |
| `ablation_s2_fine_grained_lightgbm` | LightGBM | Strategy 2 | 34-Class | 0.6218 | 72.69% | 0.6140 | 0.7175 | 69.79% | **0.6511** | 0.6876 | -0.0079 | +0.0372 |
| `ablation_s2_fine_grained_xgboost` | XGBoost | Strategy 2 | 34-Class | 0.6034 | 70.97% | 0.5993 | 0.7005 | 67.94% | 0.6351 | 0.6693 | -0.0041 | +0.0357 |
| `ablation_unweighted_fine_grained_lightgbm` | LightGBM | Unweighted | 34-Class | 0.6239 | 74.45% | **0.6220** | 0.7346 | 69.12% | 0.6217 | 0.6814 | -0.0019 | -0.0003 |
| `ablation_unweighted_fine_grained_xgboost` | XGBoost | Unweighted | 34-Class | 0.6139 | 74.03% | 0.6148 | 0.7306 | 68.22% | 0.6064 | 0.6732 | +0.0009 | -0.0084 |

---

## 4. Key Findings Supported by Exact Evidence

### Finding 1: Validation-to-Internal-Test Stability
Across all 21 models, the differences between validation Macro-F1 and internal test Macro-F1 were small:
$$\max |\Delta_{\text{gen}}| = \max |\text{F1}_{\text{test}} - \text{F1}_{\text{val}}| \le 0.0079$$
For instance, `fine_grained_lightgbm` recorded a validation Macro-F1 of **0.6142** and an internal test Macro-F1 of **0.6145** ($\Delta = +0.0003$). While these small differences demonstrate consistent performance across the internal validation and test splits, this stability reflects split consistency under identical sampling and does not prove the complete absence of overfitting.

### Finding 2: Architecture Comparisons Vary by Task Granularity
Model ranking depends on the classification task and metric under evaluation:
- **Binary Classification**: **Random Forest** achieved the highest internal test Macro-F1 (**0.7690**, Acc: 89.61%), closely followed by **LightGBM** (Macro-F1: **0.7637**, Acc: 89.15%). Both tree models outperformed linear SGD (Macro-F1: **0.6761**) and the 5-epoch PyTorch MLP (Macro-F1: **0.7125**).
- **8-Class Category Classification**: **LightGBM** achieved the highest internal test Macro-F1 (**0.6894**, Acc: 78.93%, MCC: 0.7468), ahead of Random Forest (Macro-F1: **0.6805**) and XGBoost (Macro-F1: **0.6619**).
- **34-Class Fine-Grained Classification**: **LightGBM** led balanced models on internal test Macro-F1 (**0.6145**, Acc: 72.65%, MCC: 0.7170), ahead of XGBoost (Macro-F1: **0.6001**) and Random Forest (Macro-F1: **0.5966**). Linear SGD (Macro-F1: **0.5272**) and the 5-epoch PyTorch MLP (Macro-F1: **0.5216**) achieved lower fine-grained separation.
- *Inference Profile*: Profiling in Stage 3 (`centralized_experiment_ledger.csv`) showed that LightGBM achieved single-flow P50 latency of **$220.4\ \mu\text{s}$** and batch throughput of **938,430 samples/sec**, whereas Random Forest required **$22,300\ \mu\text{s}$** single-flow latency due to joblib threadpool dispatch overhead.

### Finding 3: Precision-Recall Trade-offs in Unweighted Training
Comparing balanced and unweighted models (`fine_grained_lightgbm` vs `ablation_unweighted_fine_grained_lightgbm`) indicates distinct trade-offs:
- **Accuracy Shift**: Unweighted LightGBM yielded a higher aggregate internal test accuracy (**$74.45\%$ vs $72.65\%$**, $+1.80\%$).
- **False Negatives to Benign**: False negatives increased from **$1,517$ flows ($2.20\%$) to $3,418$ flows ($4.95\%$)**, an increase of **$+1,901$ missed attack flows (+125.3%)**.
- **Tier 4 Minority Attack Recall**: Unweighted training reduced Tier 4 Web-tier macro recall from **$34.63\%$ to $16.81\%$ (-17.82%)**:
  - `SQLINJECTION`: Recall decreased from **$35.83\% \rightarrow 7.48\%$** ($-28.35\%$)
  - `BROWSERHIJACKING`: Recall decreased from **$53.06\% \rightarrow 31.29\%$** ($-21.77\%$)
  - `DICTIONARYBRUTEFORCE`: Recall decreased from **$52.62\% \rightarrow 34.18\%$** ($-18.44\%$)
  - `UPLOADING_ATTACK`: Recall decreased from **$23.33\% \rightarrow 6.67\%$** ($-16.67\%$)
  - `XSS`: Recall decreased from **$20.22\% \rightarrow 3.83\%$** ($-16.39\%$)
- *Interpretation*: Unweighted training achieves higher overall accuracy by prioritizing high-support classes (Benign recall increased from $65.71\%$ to $83.70\%$), but substantially degrades recall on low-prevalence web-tier attacks.

### Finding 4: Observed Accuracy on Ambiguous Feature Vectors and Collision Purging
Evaluating the ambiguous validation subset ($N_{\text{val, ambig}} = 2,782$) in Stage 3:
- Models attained **approximately $41.7\%$ observed accuracy on the ambiguous validation subset**, compared to **$75.6\%$ on the common clean validation subset**. This reflects the difficulty of predicting contradictory labels for identical feature vectors, though it does not formally establish the theoretical Bayes error rate of the underlying process.
- **Strategy 2 Evaluation**: Purging collision vectors during training resulted in a small numerical increase in same-testbed temporal cross-capture Macro-F1:
  - Fine-grained LightGBM cross-capture Macro-F1 increased from **$0.6498$ (Strategy 1) to $0.6511$ (Strategy 2)**, a difference of **$+0.0013$**.
  - Internal test Macro-F1 remained comparable (**$0.6145$ vs $0.6140$**, $-0.0005$).
- *Interpretation*: The $+0.0013$ difference represents a marginal numerical improvement on temporal cross-capture evaluation, indicating that collision purging did not degrade performance and provided a slight regularization benefit on this test set.

### Finding 5: Threat-Tier Error Patterns and Web-Tier Performance
Confusion analysis in Stage 5 identified structural patterns:
1. **Intra-Tier Confusion**: **$83.3\%$** of top off-diagonal confusion pairs occurred between attack classes within the same functional threat tier (e.g., `DDOS-SYN_FLOOD` predicted as `DDOS-SYNONYMOUSIP_FLOOD` in $47.34\%$ of true cases).
2. **Reconnaissance False Negatives**: Of the 1,517 attack flows misclassified as Benign by balanced LightGBM, **$80.4\%$ ($1,220$ flows) were Tier 3 Reconnaissance attacks**, with `RECON-OSSCAN` accounting for $625$ missed flows ($21.06\%$ of that class). Volumetric floods had $< 0.1\%$ false negatives to Benign.
3. **Web-Tier Performance**: In balanced `fine_grained_lightgbm`, the **internal-test Web-tier (Tier 4) macro-F1 was 0.1748**. In the unweighted variant, it was 0.2203 (with lower recall), and under same-testbed cross-capture evaluation, balanced LightGBM reached 0.3253. Low web-tier F1 specifically describes these balanced LightGBM internal-test outcomes rather than a universal property of all models.

---

## 5. Research Contributions

### 5.1 Methodological Contributions
1. **Systematic Characterization of Redundancy**: Quantified exact duplicate rows ($5.38\%$), redundant feature vectors ($5.89\%$), and conflicting label collisions ($5.40\%$) within `data/sample_stratified.csv`, demonstrating how random splits can introduce feature leakage.
2. **Feature-Hash Partitioning Protocol**: Implemented an atomic grouping strategy that guarantees zero feature-vector overlap across train, validation, and test partitions, paired with verified preprocessors (`LeakFreePreprocessor`).
3. **Common Clean Validation Benchmark**: Constructed a dual-strategy partitioning design with an invariant clean validation set ($N = 72,390$) that allows direct comparison of collision-purging effects without partition variance.
4. **Multi-Partition Evaluation Protocol**: Evaluated models across both an internal in-distribution test set ($N = 75,105$) and a same-testbed temporal cross-capture clean test set ($N = 57,280$).

### 5.2 Empirical Findings
1. **21-Model Benchmark Ledger**: Compiled a centralized benchmark ledger (`centralized_experiment_ledger.csv`) documenting validation metrics, single-flow latency distributions, throughput, and memory consumption across 5 model families.
2. **Quantification of Class Weighting Effects**: Measured the specific trade-offs of class weighting, demonstrating that unweighted training increases aggregate accuracy ($72.65\% \rightarrow 74.45\%$) while increasing false negatives to benign traffic by $+125.3\%$ and reducing Tier 4 web-tier recall from $34.63\%$ to $16.81\%$.
3. **Marginal Impact of Collision Purging**: Documented that purging collision vectors resulted in a marginal $+0.0013$ cross-capture Macro-F1 improvement for fine-grained LightGBM, while maintaining comparable internal test performance.
4. **Threat-Tier Performance Profiling**: Benchmarked models across functional tiers, documenting strong detection on Mirai botnets ($F1 > 0.99$) and volumetric floods ($F1 > 0.79$), alongside lower performance on Tier 4 web attacks ($F1 = 0.1748$ on internal test for balanced LightGBM).

### 5.3 Standard Engineering Practices
1. **Preprocessor Serialization**: Serialized stateful transformation parameters to disk (`preprocessor_strategy1.joblib` and `preprocessor_strategy2.joblib`), ensuring that evaluation partitions are transformed without refitting.
2. **Single-Pass Evaluation**: Implemented a read-only Stage 4 runner (`src_v2/run_stage4_eval.py`) that evaluated test sets in a single pass without parameter tuning.

---

## 6. Limitations

1. **Synthetic Testbed Topology**:
   - All data was generated on a single laboratory testbed of 105 IoT devices. Traffic patterns do not capture the operational background noise, routing diversity, or protocol mixes of enterprise or service-provider networks.
2. **Scope of Temporal Cross-Capture Evaluation**:
   - The cross-capture test set (`cross_capture_test_clean.csv`, $N = 57,280$) was sampled from subsequent capture sessions (`Merged26.csv`–`Merged40.csv`) on the **same testbed using the same attack tools**. This evaluates **temporal cross-session stability**, not generalization to independent network topologies or different datasets.
3. **Sampled Class Prior in Cross-Capture Test**:
   - The cross-capture dataset capped attack classes at 2,000 samples and benign at 5,000 samples ($N = 57,280$). Because this uniform cap alters class proportions relative to the primary capture, cross-capture comparisons should rely on Macro-F1 and per-class recall rather than raw accuracy.
4. **Coupling to Standardized Features on Tree Checkpoints**:
   - Because `scale_features=True` was applied during Stage 3 data ingestion, tree models were trained on standardized inputs. Their decision thresholds reflect $z$-score coordinates, requiring standardized inputs during evaluation.
5. **Fixed-Budget MLP Training**:
   - The PyTorch MLP baseline was trained under a fixed budget of **5 epochs** (`batch_size=2048`, `lr=0.002`, `AdamW`), as verified in `src_v2/models.py` and `results_v2/metrics/*_dnn_loss.json`. More extensive hyperparameter searches or alternative architectures (e.g., TabNet, FT-Transformer) were not evaluated.
6. **Offline Microbenchmark Evaluation**:
   - Latency and throughput were benchmarked in user-space Python on a workstation CPU. These figures do not reflect in-kernel packet processing (e.g., eBPF/XDP, DPDK) or resource-constrained embedded gateway hardware.

---

## 7. Research Claims Taxonomy

```
[DEMONSTRATED EMPIRICAL FINDINGS]
 ├── Feature-hash partitioning prevents vector overlap across partitions.
 ├── Validation-to-internal-test Macro-F1 differences remain small (|Δ| ≤ 0.0079).
 ├── Random Forest leads binary internal test Macro-F1 (0.7690); LightGBM leads category (0.6894) and fine-grained (0.6145).
 ├── Unweighted training increases false negatives to Benign (+125.3%) and reduces Tier 4 recall (34.63% → 16.81%).
 ├── Models achieved ~41.7% observed accuracy on the ambiguous validation subset.
 └── Strategy 2 showed a marginal +0.0013 cross-capture Macro-F1 difference over Strategy 1.

[SUPPORTED INTERPRETATIONS]
 ├── Tree ensembles capture non-linear step thresholds in tabular flow features more readily than linear SGD.
 ├── High intra-tier confusion (e.g., DDoS vs DoS) relates to shared packet header structures differing primarily in multi-source flow dynamics.
 └── Low-rate reconnaissance probes share feature values with standard connection resets, contributing to false negatives.

[HYPOTHESES & FUTURE WORK]
 ├── Evaluation across entirely distinct IoT network topologies or real-world traffic.
 ├── Deployment and benchmarking on embedded hardware (e.g., OpenWrt gateways).
 └── Exploration of specialized architectures or representation learning for web-tier attacks.
```

---

## 8. Conclusion

This project conducted an empirical evaluation and methodological audit of the CICIoT2023 benchmark across 21 machine learning baseline configurations. The analysis established that duplicate feature vectors and label collisions are prevalent in the stratified sample, and that feature-hash partitioning effectively prevents inter-partition feature leakage.

Under this leak-free evaluation:
1. Models displayed consistent performance between validation and internal test splits ($|\Delta_{\text{gen}}| \le 0.0079$). Performance reached **$89.61\%$ accuracy / $0.7690$ Macro-F1** on Binary detection (Random Forest), **$78.93\%$ accuracy / $0.6894$ Macro-F1** on 8-Class Category classification (LightGBM), and **$72.65\%$ accuracy / $0.6145$ Macro-F1** on 34-Class Fine-Grained classification (LightGBM).
2. Architectural strengths varied: Random Forest led binary Macro-F1, while LightGBM led category and fine-grained classification with high batch throughput ($>930,000$ samples/sec).
3. Inverse-frequency class weighting provided an essential safeguard against minority attack suppression: unweighted training yielded a $+1.8\%$ accuracy increase, but cut Tier 4 web-tier recall from $34.63\%$ to $16.81\%$ and increased false negatives to benign traffic by $+125.3\%$.
4. Purging label collisions (Strategy 2) produced a marginal $+0.0013$ improvement in temporal cross-capture Macro-F1 while maintaining comparable in-distribution results.
5. Models maintained high performance on volumetric floods and Mirai botnets ($F1 > 0.79$ and $> 0.99$), while balanced LightGBM attained lower internal-test performance on Tier 4 web attacks ($F1 = 0.1748$) and experienced false negatives on OS reconnaissance probes ($21.06\%$ misclassified as Benign).

---

## 9. Document Revision Notes

This report was revised to correct several methodological claims:
1. **Literature Claims**: Removed the assertion that published results above 99% were caused by leakage. Stated only that leakage can inflate evaluation results.
2. **Overfitting**: Replaced "zero overfitting" with "small validation-to-internal-test Macro-F1 differences," clarifying that small differences do not prove the absence of overfitting.
3. **Strategy 2 Generalization**: Qualified the Strategy 2 cross-capture Macro-F1 improvement ($+0.0013$) as a marginal numerical difference rather than proof of broader generalization.
4. **Bayes Error**: Replaced theoretical "Bayes error ceiling" terminology with "approximately 41.7% observed accuracy on the ambiguous validation subset."
5. **DNN Budget**: Verified that the PyTorch Tabular DNN was trained for 5 epochs (not 15), matching `src_v2/models.py` and the saved loss histories.
6. **Model Comparisons**: Qualified comparisons by task, noting Random Forest's binary lead (0.7690) alongside LightGBM's category (0.6894) and fine-grained (0.6145) leads.
7. **Web-Tier Performance**: Clarified that $F1 \approx 0.17$ refers specifically to balanced LightGBM internal-test Tier 4 results.
8. **Collision Statistics**: Clearly distinguished 8,740 unique conflicting hashes, 28,608 associated rows, and $5.402\%$ prevalence from exact duplicate rows and identical-label vectors.
9. **Novelty Claims**: Removed "first" and "definitive proof" claims, distinguishing methodological contributions from standard engineering practices.
10. **Terminology**: Replaced "independent OOD / cross-domain generalization" with "same-testbed temporal cross-capture evaluation."
