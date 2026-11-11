# Comparative Experimental Evaluation of Hierarchical & Improved Flat 34-Class IDS Architectures (CICIoT2023)

> **Research Track**: Track 2 — Practical IDS Engineering ([`track2_ids_engineering/hierarchical_ids/`](file:///c:/Users/priya/Downloads/capstone/track2_ids_engineering/hierarchical_ids))  
> **Upstream Read-Only Reference**: Track 1 — Dataset Evaluation ([`track1_dataset_evaluation/`](file:///c:/Users/priya/Downloads/capstone/track1_dataset_evaluation))  
> **Strict Isolation Status**: All Track 1 code, splits, checkpoints, and results remain unmodified. Tuning occurred strictly on Validation; Internal Test and Cross-Capture Clean were evaluated once read-only under frozen configuration [`frozen_architectures_config.json`](file:///c:/Users/priya/Downloads/capstone/track2_ids_engineering/hierarchical_ids/results/frozen_architectures_config.json).

---

## 1. Executive Summary

This study evaluates whether hierarchical Flood vs. Non-Flood specialization improves 34-class IoT intrusion detection—especially minority-class detection—over both an existing Track 1 flat baseline (**E1**) and an equally parameterized, class-aware flat classifier (**E4**). We evaluate all four architectural families across Track 1's leakage-free Strategy 1 partitions (`Train = 350,821`, `Validation = 75,172`, `Internal Test = 75,105`) and `Cross-Capture Clean = 57,280`, including the finalized **E2 Fast Cascade (`binary_lightgbm.joblib` Stage 1)** latency fix and **E4 Merged-Flood Operational Taxonomy (29-class)** evaluation:

| Architecture Family | Model Status | Internal Test Acc | Internal Test Macro-F1 | Internal Test MCC | Test Minority F1 (Rec) | Cross Acc | Cross Macro-F1 | Cross MCC | Cross Minority F1 (Rec) | Single-Flow P50 (Test) | Model Execs / Flow | Disk Size |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E1: `E1_Flat34_LGBM_Balanced`** | Reused Track 1 | 72.65% | 0.6145 | 0.7170 | 0.1624 (**0.3360**) | 69.64% | **0.6498** | 0.6861 | **0.3142** (**0.3495**) | **3.43 ms** | 1.00 | **11.33 MB** |
| **E1-Ref: `E1_Ref_Unweighted_Track1`** | Reused Track 1 | 74.45% | 0.6220 | 0.7346 | 0.2014 (0.1531) | 69.13% | 0.6217 | 0.6814 | 0.2343 (0.1580) | 3.63 ms | 1.00 | 11.33 MB |
| **E2 Default: `E2_Cascade_Default_T050`** | Reused S1 LGBM (`0.50`) + S2/3 | 73.59% | 0.6185 | 0.7289 | 0.2115 (0.1519) | 68.61% | 0.6213 | 0.6818 | 0.2294 (0.1489) | 7.88 ms | 2.63 | 35.57 MB |
| **E2 Legacy RF: `E2_Cascade_ValTuned`** | Legacy S1 RF (`0.15`) + S2/3 | 75.04% | **0.6361** | 0.7405 | **0.2264** (0.2094) | 70.50% | 0.6453 | 0.6949 | 0.2954 (**0.2214**) | 53.30 ms* | 2.82 | 47.00 MB |
| **E2 Fast LGBM: `E2_Cascade_ValTuned_LGBM`** | **Updated S1 LGBM (`0.15`) + S2/3** | 75.13% | 0.6357 | 0.7416 | 0.2263 (0.2094) | 70.60% | 0.6444 | 0.6959 | 0.2919 (0.2175) | **8.44 ms** | 2.81 | 35.56 MB |
| **E3 Naive: `E3_Parallel_NaiveMax`** | Unsuccessful (No S2 Gate) | 56.11% | 0.4389 | 0.5567 | 0.1757 (0.1084) | 53.26% | 0.4360 | 0.5261 | 0.1707 (0.1021) | 49.73 ms* | 2.82 | 44.65 MB |
| **E3 Temp+Alpha: `E3_Parallel_TempAlpha`** | Unsuccessful (No S2 Gate) | 54.53% | 0.4301 | 0.5402 | 0.1883 (0.1208) | 51.86% | 0.4273 | 0.5108 | 0.1897 (0.1163) | 49.71 ms* | 2.82 | 44.65 MB |
| **E3 MetaGate: `E3_Parallel_MetaGate`** | Unsuccessful vs S2 Router | 73.78% | 0.6239 | 0.7275 | 0.2253 (0.2094) | 69.41% | 0.6327 | 0.6837 | 0.2948 (0.2210) | 52.40 ms* | 2.82 | 44.65 MB |
| **E4 Improved Flat: `E4_LGBM_Leaves63_Sqrt`** | **Primary Recommended Deployment** | **75.37%** | 0.6352 | **0.7440** | 0.2196 (0.2063) | **70.72%** | 0.6441 | **0.6972** | 0.2872 (0.2184) | **4.15 ms** | **1.00** | 33.90 MB |

*\*Note: Legacy `E2_Cascade_ValTuned` and all E3 architectures used Track 1's serialized [`binary_random_forest.joblib`](file:///c:/Users/priya/Downloads/capstone/track1_dataset_evaluation/models/binary_random_forest.joblib) (`n_jobs=-1`), where per-call worker thread-pool dispatch added ~45 ms per single-row query despite `8,042.5 fps` batch throughput. Replacing Stage 1 with Track 1's existing [`binary_lightgbm.joblib`](file:///c:/Users/priya/Downloads/capstone/track1_dataset_evaluation/models/binary_lightgbm.joblib) under the exact Validation-frozen thresholds (`tau_att=0.15, f_tau=0.40`) slashes E2 Test P50 latency by **6.31× (`53.30 ms -> 8.44 ms`)** and P95 by **6.28× (`64.20 ms -> 10.23 ms`)** while improving Test Accuracy (`75.04% -> 75.13%`) and MCC (`0.7405 -> 0.7416`). Even after fixing E2 latency, single-pass **`E4_LGBM_Leaves63_Sqrt`** remains faster (`4.15 ms` vs. `8.44 ms`), simpler (`1.0` vs. `2.81` model calls/flow), higher in Accuracy and MCC, and smaller (`33.90 MB` vs. `35.56 MB`), making E4 our recommended deployment IDS.*

### Key Scientific Conclusions
1. **Recommendation — Deploy `E4_LGBM_Leaves63_Sqrt`, Retain Fast `E2_Cascade_ValTuned_LGBM` as Comparative Hierarchy**: Most of E2's improvement over Track 1 **E1** (`0.6145` -> `0.6361`) came from Stage 1 threshold calibration, gentler `sqrt_balanced` weighting, and tree capacity (`num_leaves=63, n_estimators=150`) rather than cascade specialization. Once flat **E4** matches those exact settings, **E4 virtually ties E2 on Test Macro-F1** (`0.6352` vs. `0.6357` Fast E2 / `0.6361` RF E2) while **winning Test Accuracy (`75.37%`), Test MCC (`0.7440`), ROC-AUC (`0.9823`), Benign Recall among non-default models (`78.63%`), Cross Accuracy (`70.72%`), execution count (`1.0` vs. `2.81`), and single-flow latency (`4.15 ms` vs. `8.44 ms`)**. Fast E2 remains valuable only when modular policy updates require decoupling Flood and Non-Flood models.
2. **Stage 2 Routing Is Solved (`99.98%`), E3 Parallel Fusion Was Unsuccessful**: `stage2_cat7_lgbm_sqrt_balanced` commits only `50 / 75,105` Flood/Non-Flood misroutes in Test, whereas parallel specialist fusion without Stage 2 (`E3_Parallel_NaiveMax`) collapses (`0.4389` Macro-F1) because Stage 3A & 3B fire `>= 0.50` simultaneously on **76.16%** (`52,083`) of passed attacks (`26,678` cross-family errors). Even `E3_Parallel_MetaGate` (`0.6239` Test Macro-F1) leaks `966` Flood flows into Non-Flood and is strictly inferior to E2 and E4.
3. **IP-Stripped Protocol Twins Explain `42.4%` of All E4 Test Errors (Separate Operational 29-Class Taxonomy)**: Without IPs/ports/attacker-count in the 39 exported CICIoT2023 features, four protocol-identical twin groups (`SYN+SynonymousIP`, `TCP`, `UDP`, `HTTP`) cause **`7,840` of `18,501` E4 Test errors**. Grouping only these verified mitigation equivalents defines a **29-class operational mitigation taxonomy** where E4 achieves `85.81%` Accuracy, `0.6808` Macro-F1, `0.8504` MCC, and `0.9893` Flood Macro-F1 (versus `75.37%` / `0.6352` / `0.7440` on the original 34-class benchmark — a distinct operational label set, not an improvement on the 34-class task).
4. **Minority-Class Recall vs. Precision & Prior-Shift Trade-Off**: Inverse-frequency `balanced` weighting (**E1**) hits the highest Minority Recall (`0.3360` Test; `0.3495` Cross) at severe precision collapse on natural Test (`0.1624` Minority F1, e.g., `651` `VULNERABILITYSCAN` flows mislabeled `SQLINJECTION`), whereas **E4** and **E2** optimize precision-recall balance (`0.2196` / `0.2263` Test Minority F1). On Cross-Capture Clean (`5.67x` minority prior inflation via `<=2,000`/class capping), `E1`'s aggressive prior artificially yields higher precision and top Cross Macro-F1 (`0.6498`) despite lower Cross Accuracy (`69.64%` vs. `70.72%` E4).

---

## 2. Experimental Isolation, Partitions, and Class Taxonomy

### 2.1 Verified Partition Counts & Leak-Free Reuse
All models share the exact 39 numerical features (`FEATURE_COLUMNS`) and Track 1's Strategy 1 [`preprocessor_strategy1.joblib`](file:///c:/Users/priya/Downloads/capstone/track1_dataset_evaluation/models/preprocessor_strategy1.joblib) fitted exclusively on `Train`. Exact feature-vector deduplication and hash-based split disjointness (`verify_disjoint=True`) passed before training and before final evaluation:

| Split Name | Role in Track 2 Study | Total Flows | Benign | Attack (Total) | Flood (16 classes) | Non-Flood (17 classes) | Minority Attacks (8 classes) | Source File |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Train** | Fit Stage 2/3A/3B & E4 models | 350,821 | 27,855 (7.94%) | 322,966 (92.06%) | 189,288 (53.96%) | 133,678 (38.10%) | 9,108 (**2.60%**) | [`sample_stratified.csv`](file:///c:/Users/priya/Downloads/capstone/data/processed/sample_stratified.csv) (`train` via [`partition_assignments.csv`](file:///c:/Users/priya/Downloads/capstone/track1_dataset_evaluation/data/partitions/partition_assignments.csv)) |
| **Validation** | Model selection, `tau_att`, `f_tau`, `T`, MetaGate | 75,172 | 6,086 (8.10%) | 69,086 (91.90%) | 40,760 (54.22%) | 28,326 (37.68%) | 1,844 (**2.45%**) | [`sample_stratified.csv`](file:///c:/Users/priya/Downloads/capstone/data/processed/sample_stratified.csv) (`val` via [`partition_assignments.csv`](file:///c:/Users/priya/Downloads/capstone/track1_dataset_evaluation/data/partitions/partition_assignments.csv)) |
| **Internal Test** | Frozen read-only evaluation | 75,105 | 6,019 (8.01%) | 69,086 (91.99%) | 40,511 (53.94%) | 28,575 (38.05%) | 1,939 (**2.58%**) | [`sample_stratified.csv`](file:///c:/Users/priya/Downloads/capstone/data/processed/sample_stratified.csv) (`test` via [`partition_assignments.csv`](file:///c:/Users/priya/Downloads/capstone/track1_dataset_evaluation/data/partitions/partition_assignments.csv)) |
| **Cross-Capture Clean** | Same-testbed disjoint capture holdout | 57,280 | 4,982 (8.70%) | 52,298 (91.30%) | 26,247 (45.82%) | 26,051 (45.48%) | 8,384 (**14.64%**) | [`cross_capture_test_clean.csv`](file:///c:/Users/priya/Downloads/capstone/track1_dataset_evaluation/data/partitions/cross_capture_test_clean.csv) |

> **Important Methodological Note on Cross-Capture Clean**: `cross_capture_test_clean.csv` is generated from disjoint CICIoT2023 capture PCAPs collected in the **same testbed** (not an external real-world network). Furthermore, its sampling caps each class at ≤2,000 rows, raising rare/minority attack share from **2.58%** (Internal Test) to **14.64%** (**5.67×**) and elevating Non-Flood share from **38.05%** to **45.48%**.

### 2.2 Canonical Flood vs. Non-Flood Taxonomy ([`data_utils.py`](file:///c:/Users/priya/Downloads/capstone/track2_ids_engineering/hierarchical_ids/src/data_utils.py))
Matching Track 1's `FUNCTIONAL_THREAT_TIERS["Tier 1: Volumetric Floods"]` and `LABEL_MAPPING_8CLASSES`:
- **BENIGN (1 class)**: `BENIGN`.
- **Stage 3A — Flood Specialist (16 classes = `DDoS` 12 + `DoS` 4)**: `DDOS-ACK_FRAGMENTATION`, `DDOS-HTTP_FLOOD`, `DDOS-ICMP_FLOOD`, `DDOS-ICMP_FRAGMENTATION`, `DDOS-PSHACK_FLOOD`, `DDOS-RSTFINFLOOD`, `DDOS-SLOWLORIS`, `DDOS-SYNONYMOUSIP_FLOOD`, `DDOS-SYN_FLOOD`, `DDOS-TCP_FLOOD`, `DDOS-UDP_FLOOD`, `DDOS-UDP_FRAGMENTATION`, `DOS-HTTP_FLOOD`, `DOS-SYN_FLOOD`, `DOS-TCP_FLOOD`, `DOS-UDP_FLOOD`.
- **Stage 3B — Non-Flood Specialist (17 classes = `Mirai` 3 + `Recon` 5 + `Spoofing` 2 + `Web` 6 + `BruteForce` 1)**: `MIRAI-GREETH_FLOOD`, `MIRAI-GREIP_FLOOD`, `MIRAI-UDPPLAIN`, `RECON-HOSTDISCOVERY`, `RECON-OSSCAN`, `RECON-PINGSWEEP`, `RECON-PORTSCAN`, `VULNERABILITYSCAN`, `DNS_SPOOFING`, `MITM-ARPSPOOFING`, `BACKDOOR_MALWARE`, `BROWSERHIJACKING`, `COMMANDINJECTION`, `SQLINJECTION`, `UPLOADING_ATTACK`, `XSS`, `DICTIONARYBRUTEFORCE`. *(Note: Although the three Mirai classes contain `"_FLOOD"` in their label string, both Track 1 `LABEL_MAPPING_8CLASSES` and `FUNCTIONAL_THREAT_TIERS` assign `Mirai` to its own botnet family rather than Volumetric DDoS/DoS Floods.)*
- **Eight Minority Attack Classes (<1% of Train)**: `UPLOADING_ATTACK`, `RECON-PINGSWEEP`, `BACKDOOR_MALWARE`, `XSS`, `COMMANDINJECTION`, `SQLINJECTION`, `BROWSERHIJACKING`, `DICTIONARYBRUTEFORCE` (all fall inside Stage 3B Non-Flood).

---

## 3. Implemented Architectures & Mathematical Formulations

Implemented cleanly under a shared `BaseIDSArchitecture` 34-class interface in [`architectures.py`](file:///c:/Users/priya/Downloads/capstone/track2_ids_engineering/hierarchical_ids/src/architectures.py):

### E1 & E4: Flat 34-Class Classifiers (`Flat34BaselineArchitecture`)
- **E1 Baseline (`E1_Flat34_LGBM_Balanced`)**: Reuses Track 1's [`fine_grained_lightgbm.joblib`](file:///c:/Users/priya/Downloads/capstone/track1_dataset_evaluation/models/fine_grained_lightgbm.joblib) (`n_estimators=100, learning_rate=0.05, num_leaves=31`, inverse-frequency `balanced` weights). Unweighted Track 1 reference [`ablation_unweighted_fine_grained_lightgbm.joblib`](file:///c:/Users/priya/Downloads/capstone/track1_dataset_evaluation/models/ablation_unweighted_fine_grained_lightgbm.joblib) is also included as `E1_Ref_Unweighted_Track1`.
- **E4 Improved Flat (`E4_LGBM_Leaves63_Sqrt`)**: Trained on all 350,821 flows with square-root inverse class weights:
  $$w_c^{\text{sqrt}} = \sqrt{\frac{N}{K \cdot N_c}}$$
  (`n_estimators=150, learning_rate=0.05, num_leaves=63`), plus validation ablations across `log_balanced`, `clipped_balanced` ($\max w \le 10$), `XGBClassifier` (`depth=8, hist`), temperature scaling $T$, and post-hoc prior adjustment $\log P(y|x)/T - \tau_{\text{prior}} \log \pi_y$. Temperature scaling left Val Macro-F1 unchanged at argmax (`0.63576`), and positive $\tau_{\text{prior}} \in \{0.25, 0.50, 0.75\}$ raised Minority Recall up to `0.3562` at severe overall Macro-F1 degradation (`0.6300` -> `0.6074`), so uncalibrated `T=1.0` `E4_LGBM_Leaves63_Sqrt` won validation Macro-F1 selection.

### E2: Three-Stage Hierarchical Cascade (`HierarchicalCascade3Stage`)
Given feature vector $x \in \mathbb{R}^{39}$:
1. **Stage 1 (Binary Filter)**: Computes $p_{\text{att}}(x) = P_{\text{S1}}(\text{Attack} \mid x)$. If $p_{\text{att}}(x) < \tau_{\text{att}}$, exit immediately as `BENIGN` (`executions = 1`).
2. **Stage 2 (Malicious-Only 7-Category Router)**: Evaluates $P_{\text{S2}}(c \mid x, \text{Attack})$ over the 7 malicious categories $\{ \text{DDoS, DoS, Mirai, Recon, Spoofing, Web, BruteForce} \}$. Flood routing score is:
   $$p_{\text{flood}}(x) = P_{\text{S2}}(\text{DDoS} \mid x) + P_{\text{S2}}(\text{DoS} \mid x)$$
   Traffic routes to Stage 3A iff $p_{\text{flood}}(x) \ge f_\tau$ (when tuned, else category argmax in $\{\text{DDoS}, \text{DoS}\}$).
   *(When reusing Track 1's 8-class Stage 2 `category_lightgbm.joblib`, invalid `Benign` predictions on Stage 1 attacks—observed on `2,393` true Validation attacks—are prevented by `cat7_attack` training strictly on malicious samples).*
3. **Stage 3A (16-Class Flood Specialist)** or **Stage 3B (17-Class Non-Flood Specialist)**: Emits fine-grained conditional distribution inside global 34-class indices (`executions = 3`).
- **Frozen Configurations**:
  - `E2_Cascade_Default_T050`: Reused Stage 1 `binary_lightgbm.joblib` ($\tau_{\text{att}} = 0.50$), `stage2_cat7_lgbm_balanced` (argmax routing), `stage3a_flood_lgbm_d63_sqrt`, `stage3b_nonflood_lgbm_d63_sqrt`.
  - `E2_Cascade_ValTuned`: Validation-best `binary_random_forest.joblib` ($\tau_{\text{att}} = 0.15$), `stage2_cat7_lgbm_sqrt_balanced` ($f_\tau = 0.40$), `stage3a_flood_lgbm_d63_sqrt`, `stage3b_nonflood_lgbm_d63_sqrt` (`n_estimators=150, num_leaves=63, learning_rate=0.05, sqrt_balanced`).

### E3: Parallel Specialist Score-Fusion Architectures (`ParallelSpecialistFusion`)
Stage 1 runs first (`tau_att = 0.15`). Passed flows invoke **both** Stage 3A ($P_{3A}(y \mid x, \text{Flood})$) and Stage 3B ($P_{3B}(y \mid x, \text{NonFlood})$) in parallel (`executions = 3`):
- **`E3_Parallel_NaiveMax`**: Stacks raw $[P_{3A}, P_{3B}]$, normalizes over 33 classes, takes global argmax.
- **`E3_Parallel_TempAlpha`**: Applies validation NLL-fitted temperatures ($T_{3A} = 1.1742$, $T_{3B} = 0.9042$) and static mixing weight $\alpha = 0.50$: $[\alpha \tilde{P}_{3A}, (1-\alpha)\tilde{P}_{3B}]$.
- **`E3_Parallel_MetaGate`**: Feeds temperature-calibrated $\tilde{P}_{3A}$, $\tilde{P}_{3B}$, and 7 output summary statistics (log max confidence, max diff, top-2 probability margins, normalized Shannon entropies = 40 meta-features, zero raw packet features) into a `LogisticRegression` gate [`e3_meta_gate_lr.joblib`](file:///c:/Users/priya/Downloads/capstone/track2_ids_engineering/hierarchical_ids/models/e3_meta_gate_lr.joblib) predicting $g(x) = P(\text{Flood} \mid \tilde{P}_{3A}, \tilde{P}_{3B})$, outputting $[g(x)\tilde{P}_{3A}, (1 - g(x))\tilde{P}_{3B}]$.

---

## 4. Stage Component Benchmarks & Validation Tuning

From [`val_component_benchmarks.csv`](file:///c:/Users/priya/Downloads/capstone/track2_ids_engineering/hierarchical_ids/results/val_component_benchmarks.csv) on `Validation (N = 75,172)`:

| Stage | Candidate ID | Weights / Origin | Val Samples | Val Acc | Key Stage Metric |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Stage 1 Binary** | `binary_lightgbm_reused` | `balanced` (Track 1) | 75,172 | 89.28% | Attack Rec = `0.8857`, Benign Rec = `0.9727` (`@0.50`) |
| **Stage 1 Binary** | `binary_random_forest_reused` | `balanced` (Track 1) | 75,172 | 89.82% | Attack Rec = `0.8924`, Benign Rec = `0.9637` (`@0.50`) |
| **Stage 2 Category** | `stage2_cat7_lgbm_balanced` | `balanced` (malicious-only) | 69,086 | 80.48% | Flood Route F1 = `0.99939`, Rec = `0.99934` |
| **Stage 2 Category** | **`stage2_cat7_lgbm_sqrt_balanced`** | `sqrt_balanced` (malicious-only) | 69,086 | **86.47%** | Flood Route F1 = **`0.99942`**, Rec = **`0.99961`** |
| **Stage 2 Category** | `stage2_cat8_lgbm_reused` | `balanced` (Track 1) | 69,086 | 78.88% | Invalidly predicted `Benign` on `2,393` true attacks |
| **Stage 3A (16 Flood)** | `stage3a_flood_lgbm_balanced` | `balanced` (`L=31, n=100`) | 40,760 | 79.61% | Flood Macro-F1 = `0.79337` (NLL `0.38628`) |
| **Stage 3A (16 Flood)** | **`stage3a_flood_lgbm_d63_sqrt`** | `sqrt_balanced` (`L=63, n=150`) | 40,760 | **79.81%** | Flood Macro-F1 = **`0.79682`** (NLL `0.38474`) |
| **Stage 3A (16 Flood)** | `stage3a_flood_xgb_sqrt` | `sqrt_balanced` (`depth=6, n=100`) | 40,760 | 79.10% | Flood Macro-F1 = `0.78577` (NLL `0.42167`) |
| **Stage 3B (17 NonFlood)** | `stage3b_nonflood_lgbm_balanced` | `balanced` (`L=31, n=100`) | 28,326 | 65.50% | Non-Flood Macro-F1 = `0.46451` (NLL `0.96210`) |
| **Stage 3B (17 NonFlood)** | **`stage3b_nonflood_lgbm_d63_sqrt`** | `sqrt_balanced` (`L=63, n=150`) | 28,326 | **71.56%** | Non-Flood Macro-F1 = **`0.51033`** (NLL `0.78037`) |
| **Stage 3B (17 NonFlood)** | `stage3b_nonflood_xgb_sqrt` | `sqrt_balanced` (`depth=6, n=100`) | 28,326 | 69.21% | Non-Flood Macro-F1 = `0.49376` (NLL `0.91233`) |

---

## 5. End-to-End Quantitative Results Across All Three Splits

Exact metrics verified from [`final_comparison_val.csv`](file:///c:/Users/priya/Downloads/capstone/track2_ids_engineering/hierarchical_ids/results/final_comparison_val.csv), [`final_comparison_test.csv`](file:///c:/Users/priya/Downloads/capstone/track2_ids_engineering/hierarchical_ids/results/final_comparison_test.csv), and [`final_comparison_cross.csv`](file:///c:/Users/priya/Downloads/capstone/track2_ids_engineering/hierarchical_ids/results/final_comparison_cross.csv):

### 5.1 Validation Split (`N = 75,172`, Read-Only Re-Evaluation Under Frozen Params)
| Architecture | Acc | Macro-F1 | Weighted-F1 | MCC | Flood F1 (Rec) | Non-Flood F1 (Rec) | Minority F1 (Rec) | Benign Rec | Benign FPR | Attack->Benign FNR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `E1_Flat34_LGBM_Balanced` | 0.72455 | 0.61418 | 0.73206 | 0.71492 | 0.79307 (0.79907) | 0.44146 (0.51164) | 0.16560 (**0.35127**) | 0.65840 | 0.34160 | **0.02245** |
| `E2_Cascade_Default_T050` | 0.73521 | 0.61958 | 0.72059 | 0.72807 | 0.79617 (0.80014) | 0.45483 (0.40186) | 0.21430 (0.15237) | **0.97272** | **0.02728** | 0.11429 |
| `E2_Cascade_ValTuned` *(Legacy RF S1)* | 0.74914 | **0.63944** | 0.74362 | 0.73924 | **0.79621** (**0.80056**) | **0.48832** (0.48070) | **0.24264** (0.22528) | 0.74203 | 0.25797 | 0.03325 |
| `E2_Cascade_ValTuned_LGBM` *(Fast S1, P50=8.44ms)* | 0.75064 | 0.63811 | 0.74399 | 0.74082 | **0.79621** (0.80054) | 0.48504 (0.47499) | 0.23675 (0.21704) | 0.77917 | 0.22083 | 0.03640 |
| `E3_Parallel_NaiveMax` | 0.55969 | 0.43886 | 0.52564 | 0.55565 | 0.56599 (0.62099) | 0.30383 (0.29148) | 0.17535 (0.10958) | 0.74203 | 0.25797 | 0.03325 |
| `E3_Parallel_TempAlpha` | 0.54504 | 0.43244 | 0.51466 | 0.54032 | 0.53878 (0.56862) | 0.31661 (0.31755) | 0.19027 (0.12293) | 0.74203 | 0.25797 | 0.03325 |
| `E3_Parallel_MetaGate` | 0.73590 | 0.62668 | 0.72708 | 0.72557 | 0.77924 (0.77920) | 0.47877 (0.47971) | 0.24248 (0.22528) | 0.74203 | 0.25797 | 0.03325 |
| `E4_LGBM_Leaves63_Sqrt` | **0.75133** | 0.63576 | **0.74457** | **0.74157** | 0.79479 (0.79943) | 0.48120 (0.47680) | 0.22709 (0.21942) | 0.78426 | 0.21574 | 0.03507 |
| `E1_Ref_Unweighted_Track1` | 0.74334 | 0.62392 | 0.73295 | 0.73332 | 0.78766 (0.78982) | 0.46505 (0.44282) | 0.21326 (0.16887) | 0.84637 | 0.15363 | 0.04887 |

### 5.2 Internal Test Split (`N = 75,105`, Frozen Read-Only)
| Architecture | Acc | Macro-F1 | Weighted-F1 | MCC | Macro AP | ROC-AUC | Flood F1 | NonFlood F1 | Minority F1 (Rec) | Benign Rec (FPR) | Att->Ben FNR | P50 (us) | P95 (us) | Batch (fps) | Size (MB) | RSS (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `E1_Flat34_LGBM_Balanced` | 0.72653 | 0.61452 | 0.73486 | 0.71703 | 0.66186 | 0.98112 | 0.79430 | 0.44098 | 0.16241 (**0.33600**) | 0.65709 (0.34291) | **0.02196** | **3,426.7** | **4,281.4** | 7,202.6 | **11.331** | **566.2** |
| `E2_Cascade_Default_T050` | 0.73589 | 0.61845 | 0.72132 | 0.72895 | 0.64422 | 0.97187 | **0.79714** | 0.45195 | 0.21152 (0.15191) | **0.97375** (**0.02625**) | 0.11565 | 7,884.1 | 9,167.3 | 8,329.6 | 35.566 | 576.6 |
| `E2_Cascade_ValTuned` *(Legacy RF S1)* | 0.75035 | **0.63614** | 0.74523 | 0.74051 | 0.66220 | 0.97578 | 0.79708 | **0.48104** | **0.22641** (0.20936) | 0.73866 (0.26134) | 0.03294 | 53,297.8 | 64,204.1 | 8,042.5 | 46.996 | 585.8 |
| `E2_Cascade_ValTuned_LGBM` *(Fast S1)* | 0.75133 | 0.63573 | 0.74505 | 0.74155 | 0.66225 | 0.97612 | 0.79711 | 0.47960 | 0.22633 (0.20937) | 0.77521 (0.22479) | 0.03611 | 8,440.3 | 10,229.0 | 7,527.6 | 35.561 | 579.8 |
| `E3_Parallel_NaiveMax` | 0.56109 | 0.43890 | 0.52623 | 0.55671 | 0.61003 | 0.96278 | 0.56739 | 0.30273 | 0.17568 (0.10839) | 0.73866 (0.26134) | 0.03294 | 49,734.2 | 56,260.3 | 4,423.6 | 44.651 | 600.3 |
| `E3_Parallel_TempAlpha` | 0.54529 | 0.43006 | 0.51344 | 0.54022 | 0.61184 | 0.96436 | 0.53684 | 0.31381 | 0.18832 (0.12081) | 0.73866 (0.26134) | 0.03294 | 49,713.1 | 57,852.1 | 6,068.8 | 44.651 | 613.1 |
| `E3_Parallel_MetaGate` | 0.73775 | 0.62392 | 0.72976 | 0.72749 | 0.65613 | 0.97612 | 0.78094 | 0.47179 | 0.22529 (0.20936) | 0.73866 (0.26134) | 0.03294 | 52,403.1 | 67,203.0 | 6,016.5 | 44.652 | 627.5 |
| `E4_LGBM_Leaves63_Sqrt` | **0.75366** | 0.63523 | **0.74721** | **0.74402** | **0.66958** | **0.98233** | 0.79603 | 0.47885 | 0.21957 (0.20631) | 0.78634 (0.21366) | 0.03444 | 4,154.3 | 6,989.3 | 5,193.1 | 33.895 | 648.1 |
| `E1_Ref_Unweighted_Track1` | 0.74452 | 0.62202 | 0.73464 | 0.73456 | 0.64313 | 0.97118 | 0.79008 | 0.45949 | 0.20137 (0.15305) | 0.83702 (0.16298) | 0.04947 | 3,631.0 | 6,797.1 | **12,030.1** | 11.327 | 667.9 |

### 5.3 Cross-Capture Clean Holdout (`N = 57,280`, Frozen Read-Only, Prior-Shifted Same-Testbed Captures)
| Architecture | Acc | Macro-F1 | Weighted-F1 | MCC | Macro AP | ROC-AUC | Flood F1 | NonFlood F1 | Minority F1 (Rec) | Benign Rec (FPR) | Att->Ben FNR | P50 (us) | P95 (us) | Batch (fps) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `E1_Flat34_LGBM_Balanced` | 0.69640 | **0.64981** | **0.69891** | 0.68606 | 0.68815 | 0.97619 | 0.80840 | **0.49940** | **0.31419** (**0.34952**) | 0.65516 (0.34484) | **0.02882** | **3,844.0** | 7,237.4 | 10,158.4 |
| `E2_Cascade_Default_T050` | 0.68605 | 0.62134 | 0.66757 | 0.68180 | 0.65730 | 0.96031 | 0.80873 | 0.45176 | 0.22936 (0.14888) | **0.97310** (**0.02690**) | 0.17838 | 8,364.5 | 12,981.7 | **12,836.5** |
| `E2_Cascade_ValTuned` *(Legacy RF S1)* | 0.70503 | 0.64531 | 0.69659 | 0.69493 | 0.68270 | 0.96866 | 0.80863 | 0.49066 | 0.29537 (0.22137) | 0.74528 (0.25472) | 0.04849 | 57,609.6 | 73,203.7 | 7,958.2 |
| `E2_Cascade_ValTuned_LGBM` *(Fast S1)* | 0.70597 | 0.64441 | 0.69612 | 0.69590 | 0.68249 | 0.96861 | 0.80863 | 0.48843 | 0.29191 (0.21748) | 0.77840 (0.22160) | 0.05241 | 7,098.7 | 9,172.9 | 11,380.6 |
| `E3_Parallel_NaiveMax` | 0.53259 | 0.43596 | 0.48947 | 0.52610 | 0.62860 | 0.95751 | 0.56835 | 0.29812 | 0.17073 (0.10212) | 0.74528 (0.25472) | 0.04849 | 66,410.2 | 84,459.7 | 5,098.8 |
| `E3_Parallel_TempAlpha` | 0.51856 | 0.42731 | 0.48040 | 0.51075 | 0.63004 | 0.95900 | 0.53714 | 0.31019 | 0.18972 (0.11632) | 0.74528 (0.25472) | 0.04849 | 81,657.0 | 102,404.2 | 4,060.4 |
| `E3_Parallel_MetaGate` | 0.69413 | 0.63271 | 0.68348 | 0.68365 | 0.67556 | 0.96911 | 0.79143 | 0.48166 | 0.29484 (0.22097) | 0.74528 (0.25472) | 0.04849 | 72,422.4 | 85,769.7 | 4,564.4 |
| `E4_LGBM_Leaves63_Sqrt` | **0.70721** | 0.64405 | 0.69703 | **0.69715** | **0.68907** | **0.97667** | **0.80892** | 0.48696 | 0.28715 (0.21835) | 0.78322 (0.21678) | 0.05059 | 6,524.2 | 8,311.8 | 4,068.8 |
| `E1_Ref_Unweighted_Track1` | 0.69125 | 0.62172 | 0.67506 | 0.68137 | 0.65888 | 0.96524 | 0.80202 | 0.45131 | 0.23432 (0.15799) | 0.83741 (0.16259) | 0.07668 | 5,117.7 | **6,309.4** | 10,722.4 |

---

## 6. Per-Class Minority Attack Performance (`test` vs. `cross`)

Exact class-level Recall / F1 from [`minority_attack_comparison_test.csv`](file:///c:/Users/priya/Downloads/capstone/track2_ids_engineering/hierarchical_ids/results/minority_attack_comparison_test.csv) and [`minority_attack_comparison_cross.csv`](file:///c:/Users/priya/Downloads/capstone/track2_ids_engineering/hierarchical_ids/results/minority_attack_comparison_cross.csv):

### Internal Test Minority Recall / F1 (`N = 1,939` minority flows across 8 classes)
| Class (Support) | E1 (`LGBM Bal`) Rec / F1 | E2 Default Rec / F1 | E2 ValTuned Rec / F1 | E3 MetaGate Rec / F1 | E4 (`Leaves63 Sqrt`) Rec / F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `UPLOADING_ATTACK` (60) | **0.2333** / **0.0545** | 0.0167 / 0.0270 | 0.0167 / 0.0220 | 0.0167 / 0.0220 | 0.0333 / 0.0364 |
| `RECON-PINGSWEEP` (125) | **0.2640** / 0.0754 | 0.1040 / 0.1595 | 0.1520 / **0.1801** | 0.1520 / **0.1801** | 0.0960 / 0.1127 |
| `BACKDOOR_MALWARE` (139) | **0.2518** / 0.0968 | 0.1079 / 0.1376 | 0.1295 / 0.1369 | 0.1295 / 0.1369 | 0.1511 / **0.1433** |
| `XSS` (183) | **0.2022** / 0.0715 | 0.0656 / 0.0886 | 0.0874 / 0.0930 | 0.0874 / 0.0930 | 0.0874 / **0.0997** |
| `COMMANDINJECTION` (255) | 0.3216 / 0.2587 | 0.2706 / **0.3503** | **0.3412** / 0.3446 | **0.3412** / 0.3446 | 0.3333 / 0.3353 |
| `SQLINJECTION` (254) | **0.3583** / 0.0834 | 0.0827 / 0.1317 | 0.1102 / **0.1458** | 0.1102 / 0.1451 | 0.1024 / 0.1244 |
| `BROWSERHIJACKING` (294) | **0.5306** / 0.2870 | 0.2483 / 0.3596 | 0.3912 / 0.4389 | 0.3912 / 0.4307 | 0.4082 / **0.4436** |
| `DICTIONARYBRUTEFORCE` (629) | **0.5262** / 0.3719 | 0.3196 / 0.4379 | 0.4467 / 0.4500 | 0.4467 / 0.4500 | 0.4388 / **0.4612** |
| **8-Class Minority Mean** | **0.3360** / 0.1624 | 0.1519 / 0.2115 | 0.2094 / **0.2264** | 0.2094 / 0.2253 | 0.2063 / 0.2196 |

### Cross-Capture Minority Recall / F1 (`N = 8,384` minority flows — 5.67× natural Test prevalence)
| Class (Support) | E1 (`LGBM Bal`) Rec / F1 | E2 ValTuned Rec / F1 | E3 MetaGate Rec / F1 | E4 (`Leaves63 Sqrt`) Rec / F1 |
| :--- | :---: | :---: | :---: | :---: |
| `UPLOADING_ATTACK` (289) | **0.2076** / **0.1135** | 0.0623 / 0.0928 | 0.0623 / 0.0928 | 0.0796 / 0.1055 |
| `RECON-PINGSWEEP` (567) | **0.3104** / **0.2366** | 0.1252 / 0.1970 | 0.1252 / 0.1970 | 0.1005 / 0.1551 |
| `BACKDOOR_MALWARE` (729) | **0.2689** / **0.2197** | 0.1536 / 0.2024 | 0.1523 / 0.2007 | 0.1564 / 0.2025 |
| `XSS` (925) | **0.2670** / **0.2138** | 0.1373 / 0.1801 | 0.1362 / 0.1789 | 0.1135 / 0.1553 |
| `COMMANDINJECTION` (1,322) | 0.3396 / 0.3975 | 0.3366 / 0.4214 | 0.3366 / 0.4214 | **0.3480** / **0.4289** |
| `SQLINJECTION` (1,233) | **0.3690** / **0.2846** | 0.1144 / 0.1934 | 0.1135 / 0.1922 | 0.1168 / 0.1926 |
| `BROWSERHIJACKING` (1,320) | **0.5083** / 0.5101 | 0.3864 / **0.5282** | 0.3864 / 0.5280 | 0.3803 / 0.5141 |
| `DICTIONARYBRUTEFORCE` (1,999) | **0.5253** / 0.5378 | 0.4552 / 0.5477 | 0.4552 / **0.5479** | 0.4517 / 0.5432 |
| **8-Class Minority Mean** | **0.3495** / **0.3142** | 0.2214 / 0.2954 | 0.2210 / 0.2948 | 0.2184 / 0.2872 |

*(Note: Updated fast `E2_Cascade_ValTuned_LGBM` in [`e2_fast_lgbm_minority_per_class.csv`](file:///c:/Users/priya/Downloads/capstone/track2_ids_engineering/hierarchical_ids/results/e2_fast_lgbm_minority_per_class.csv) maintains the same Minority Mean Recall / F1: **0.2094 / 0.2263** on Test and **0.2175 / 0.2919** on Cross.)*

---

## 7. Stage-by-Stage Error Attribution & Diagnostic Analysis

### 7.1 Full Error Decomposition Across Splits ([`error_attribution_test.csv`](file:///c:/Users/priya/Downloads/capstone/track2_ids_engineering/hierarchical_ids/results/error_attribution_test.csv), [`error_attribution_cross.csv`](file:///c:/Users/priya/Downloads/capstone/track2_ids_engineering/hierarchical_ids/results/error_attribution_cross.csv))
Every misclassified flow falls into one of six mutually exclusive buckets:

| Split & Architecture | Total Errors | Attack->Benign (FN) | Benign->Attack (FP) | Flood -> Non-Flood | Non-Flood -> Flood | Within-Flood Confusion | Within-NonFlood Confusion |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Test** `E1_Flat34_LGBM_Balanced` | 20,539 | **1,517** | 2,064 | 16 | 34 | 8,134 | 8,774 |
| **Test** `E2_Cascade_Default_T050` | 19,836 | 7,990 | **158** | 21 | 29 | **8,052** | **3,586** *(masked by FN)* |
| **Test** `E2_Cascade_ValTuned` *(Legacy RF S1)* | 18,750 | 2,276 | 1,573 | **9** | 41 | 8,055 | 6,796 |
| **Test** `E2_Cascade_ValTuned_LGBM` *(Fast S1)* | 18,676 | 2,495 | 1,353 | **9** | 41 | 8,055 | 6,723 |
| **Test** `E3_Parallel_NaiveMax` | 32,964 | 2,276 | 1,573 | 13,347 | 13,331 | 1,866 | 571 |
| **Test** `E3_Parallel_MetaGate` | 19,696 | 2,276 | 1,573 | 966 | 87 | 8,005 | 6,789 |
| **Test** `E4_LGBM_Leaves63_Sqrt` | **18,501** | 2,379 | 1,286 | 11 | **35** | 8,087 | 6,703 |
| **Cross** `E1_Flat34_LGBM_Balanced` | 17,390 | **1,507** | 1,718 | 19 | 15 | 4,388 | 9,743 |
| **Cross** `E2_Cascade_Default_T050` | 17,983 | 9,329 | **134** | 30 | **14** | 4,367 | 4,109 |
| **Cross** `E2_Cascade_ValTuned` *(Legacy RF S1)* | 16,896 | 2,536 | 1,269 | **11** | 19 | 4,377 | 8,684 |
| **Cross** `E2_Cascade_ValTuned_LGBM` *(Fast S1)* | 16,842 | 2,741 | 1,104 | **11** | 19 | 4,377 | 8,590 |
| **Cross** `E3_Parallel_NaiveMax` | 26,773 | 2,536 | 1,269 | 7,473 | 13,813 | 1,138 | 544 |
| **Cross** `E3_Parallel_MetaGate` | 17,520 | 2,536 | 1,269 | 653 | 44 | 4,340 | 8,678 |
| **Cross** `E4_LGBM_Leaves63_Sqrt` | **16,771** | 2,646 | **1,080** | **11** | 16 | 4,373 | **8,645** |

### 7.2 Answers to Core Architectural Questions (Experimental Evidence)

1. **Why did Stage 1 fail so heavily at threshold `0.50` (`E2_Cascade_Default_T050`)?**
   Track 1 trained `binary_lightgbm.joblib` with `class_weight="balanced"`. Because Attacks outnumber Benign `11.6:1` in Train (`322,966 : 27,855`), each Benign sample received `~11.6×` weight, making the default `0.50` boundary overly conservative toward Benign: `97.38%` Test Benign Recall came at the cost of exiting **`7,990` (11.57%)** true Test attacks and **`9,329` (17.84%)** true Cross attacks at Stage 1 before Stage 2 ever saw them. Tuning `tau_att = 0.15` reduced missed attacks to `2,276` RF / `2,495` LGBM (Test) and `2,536` RF / `2,741` LGBM (Cross), raising Test Macro-F1 by `+0.0173` (`0.6185 -> 0.6357` LGBM), while raising Benign FPR (`2.63% -> 22.48%` LGBM).
2. **Did replacing E2's Stage 1 Random Forest with LightGBM solve the latency bottleneck?**
   **Yes.** Legacy `E2_Cascade_ValTuned` suffered `53.30 ms` (`53,297.8 us`) Test P50 single-flow latency solely from scikit-learn `n_jobs=-1` worker thread-pool dispatch in `binary_random_forest.joblib`. Swapping in Track 1's existing [`binary_lightgbm.joblib`](file:///c:/Users/priya/Downloads/capstone/track1_dataset_evaluation/models/binary_lightgbm.joblib) under the already Validation-selected settings (`tau_att=0.15, f_tau=0.40`) preserves Validation Macro-F1 (`0.63811`), **cuts Test P50 latency by 6.31× to `8.44 ms`** (`P95 = 10.23 ms`, batch `7,527.6 fps`), and actually improves Test Accuracy (`75.04% -> 75.13%`), Benign Recall (`73.87% -> 77.52%`), and MCC (`0.7405 -> 0.7416`). Even with this fix, single-pass E4 remains over **2.0× faster** (`4.15 ms` vs. `8.44 ms`) at single-flow inference.
3. **Is Stage 2 routing a bottleneck in E2?**
   **No.** On Test (`stage_diagnostics_test.json`), `stage2_cat7_lgbm_sqrt_balanced` achieved **`99.98%` Flood Recall / `99.90%` Flood Precision**, committing only **50 total Flood/Non-Flood misroutes out of 75,105 flows** (`9` Flood->NonFlood + `41` NonFlood->Flood) on Test, and only **30 out of 57,280** (`11 + 19`) on Cross. Training Stage 2 strictly on the 7 attack categories eliminated all `2,393` invalid Stage 2 Benign exits observed with Track 1's 8-class router.
4. **Which traffic family benefited from specialization (`E2` vs. `E1`), and did `E4` close the gap?**
   - On Internal Test (`E2` vs. `E1`): **Non-Flood** Macro-F1 rose from `0.4410` to **`0.4810`** (`0.4796` Fast E2), whereas **Flood** rose only `0.7943 -> 0.7971`.
   - Comparing **`E2_Cascade_ValTuned_LGBM` vs. capacity/weight-matched `E4_LGBM_Leaves63_Sqrt`**: E4 matches Flood (`0.7960` vs. `0.7971` Test, `0.8089` vs. `0.8086` Cross), virtually ties Non-Flood (`0.4789` vs. `0.4796` Test), and outperforms Fast E2 in Accuracy (`75.37%` vs. `75.13%`), MCC (`0.7440` vs. `0.7416`), ROC-AUC (`0.9823` vs. `0.9761`), Benign Recall (`78.63%` vs. `77.52%`), and P50 latency (`4.15 ms` vs. `8.44 ms`).
5. **Why was E3 Parallel Fusion unsuccessful?**
   Conditionally trained specialists (`Stage 3A` on Floods, `Stage 3B` on Non-Floods) never learn out-of-distribution rejection: **`76.16%` (`52,083` Test)** of passed attacks triggered `>= 0.50` probability in both 3A and 3B simultaneously, causing `E3_Parallel_NaiveMax` to make **`26,678` cross-family errors** (`56.11%` Acc / `0.4389` Macro-F1). Static scaling (`E3_Parallel_TempAlpha`, `0.4301` F1) did not resolve OOD overconfidence. Adding `E3_Parallel_MetaGate` recovered Macro-F1 to `0.6239` Test (`0.6327` Cross) yet still leaked `966` Flood flows into Non-Flood (`DOS-UDP_FLOOD` F1 fell from `0.6778` to `0.4573`), remaining strictly worse than E2 and E4.

### 7.3 Merged-Flood Operational Taxonomy Investigation & Recalculated E4 Results ([`merged_flood_e4_summary.csv`](file:///c:/Users/priya/Downloads/capstone/track2_ids_engineering/hierarchical_ids/results/merged_flood_e4_summary.csv))

> **Methodological Warning (Distinct Tasks)**: Merging Flood protocol twins defines an **alternative 29-class operational mitigation taxonomy (`OP29_MERGED_FLOOD`)** for L4/L7 firewall policy enforcement. It MUST **NOT** be reported as improving upon E4's original **34-class** classification score (`75.37%` Acc / `0.6352` Macro-F1).

Direct schema inspection of raw [`Merged01.csv`](file:///c:/Users/priya/Downloads/capstone/data/raw/Merged01.csv) confirms **0 of 40 columns** record IP addresses, ports, timestamps, or distinct source host counts. Auditing E4's exact 34×34 Test confusion matrix ([`merged_flood_twin_group_audit.csv`](file:///c:/Users/priya/Downloads/capstone/track2_ids_engineering/hierarchical_ids/results/merged_flood_twin_group_audit.csv)) verifies that **`7,840` of all `18,501` E4 Test errors (`42.4%`)** are swaps strictly *inside* four protocol-identical twin groups while every remaining Flood class (`RSTFINFLOOD`, `ICMP_FLOOD`, `PSHACK`, `ACK/ICMP/UDP Fragmentation`, `SLOWLORIS`) already achieves `0.933–0.9998` F1:

| Group | Merged Classes (`OP29` Label) | Train Shared Vector Hashes | Test Support | Pre-Merge Swaps | In-Group Retention | Mean 34-Cls F1 -> Merged `OP29` F1 | Justified Mitigation Equivalence |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`SYN_TRIO`** | `DDOS-SYN_FLOOD`, `DOS-SYN_FLOOD`, `DDOS-SYNONYMOUSIP_FLOOD` -> `FLOOD_SYN` | `3,072` | 7,844 | **4,118** | **99.62%** *(only 30 leaked)* | `0.4626` -> **`0.9975`** | Identical bare TCP SYN flow stats (`syn=1, ack=0, size=60`); mitigated via **SYN Cookies + half-open rate limit** (+ stateless L3 `src==dst` ACL where IPs exist). |
| *`SYN_PAIR` Check* | *If `DDOS-SYNONYMOUSIP_FLOOD` kept separate (`OP30`, 30-cls)* | `1,339` | 5,385 | 1,028 | *56.90% (2,321 leaked!)* | Sensitivity check only | Proves empirically why omitting `SYNONYMOUSIP` leaves `2,296` SYN flows colliding into `SYNONYMOUSIP`. |
| **`TCP_PAIR`** | `DDOS-TCP_FLOOD`, `DOS-TCP_FLOOD` -> `FLOOD_TCP` | `1,913` | 5,078 | **1,817** | **99.80%** | `0.6333` -> **`0.9989`** | Volumetric TCP FlowSpec drop & state scrub (DDoS vs. DoS indistinguishable without source IP cardinality). |
| **`UDP_PAIR`** | `DDOS-UDP_FLOOD`, `DOS-UDP_FLOOD` -> `FLOOD_UDP` | `958` | 5,357 | **1,536** | **99.68%** | `0.7071` -> **`0.9974`** | Upstream UDP policing & QoS ingress rate-limiting. |
| **`HTTP_PAIR`** | `DDOS-HTTP_FLOOD`, `DOS-HTTP_FLOOD` -> `FLOOD_HTTP` | `1` *(continuous rates)* | 4,244 | **369** | **98.75%** | `0.8825` -> **`0.9858`** | L7 WAF HTTP request challenge & reverse-proxy connection throttle. |

Recalculating frozen `E4_LGBM_Leaves63_Sqrt` across the **exact same test samples (`N = 75,105`)** under both tasks (and cross-checking Validation `N = 75,172` and Cross-Capture `N = 57,280`):

| Split & Taxonomy (Evaluated on Frozen E4 Outputs) | Classes | Acc | Macro-F1 | Weighted-F1 | MCC | Flood Macro-F1 | Non-Flood Macro-F1 | Minority Macro-F1 (Rec) | Benign F1 (Rec) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Test** `ORIG34` *(Academic Standard Benchmark)* | 34 | 0.75366 | 0.63523 | 0.74721 | 0.74402 | 0.79603 | 0.47885 | 0.21957 (0.20631) | 0.72089 (0.78634) |
| **Test** `OP30_STRICT_4_PAIR` *(Sensitivity: SynIP Unmerged)* | 30 | 0.81691 | 0.66421 | 0.81366 | 0.80834 | 0.92206 | 0.47885 | 0.21957 (0.20631) | 0.72089 (0.78634) |
| **Test** `OP29_MERGED_FLOOD` *(`label_mapped` -> Collapsed CM)* | **29** | **0.85805** | **0.68082** | **0.85341** | **0.85038** | **0.98931** | **0.47885** *(unchanged)* | **0.21957** *(0.20631)* | **0.72089** *(0.78634)* |
| **Test** `OP29_MERGED_FLOOD` *(`prob_sum_argmax`)* | 29 | 0.85809 | 0.68088 | 0.85345 | 0.85042 | 0.98948 | 0.47885 | 0.21957 (0.20631) | 0.72089 (0.78634) |
| **Val** `OP29_MERGED_FLOOD` *(`label_mapped`, N=75,172)* | 29 | 0.85822 | 0.68244 | 0.85350 | 0.85048 | 0.99015 | 0.48120 | 0.22709 (0.21942) | 0.71872 (0.78426) |
| **Cross** `OP29_MERGED_FLOOD` *(`label_mapped`, N=57,280)* | 29 | 0.78036 | 0.68527 | 0.77081 | 0.77123 | 0.99253 | 0.48696 | 0.28715 (0.21835) | 0.67684 (0.78322) |

Non-Flood (`0.47885`) and Minority (`0.21957` F1 / `0.20631` Rec) metrics are bit-identical between `ORIG34` and `OP29`—confirming all improvement lies strictly in eliminating IP-less flood twin swaps. Visual comparisons are saved in [`results/figures/e1_e4_architecture_tradeoffs.png`](file:///c:/Users/priya/Downloads/capstone/track2_ids_engineering/hierarchical_ids/results/figures/e1_e4_architecture_tradeoffs.png) and [`results/figures/e4_34class_vs_merged_flood.png`](file:///c:/Users/priya/Downloads/capstone/track2_ids_engineering/hierarchical_ids/results/figures/e4_34class_vs_merged_flood.png).

---

## 8. Limitations, Deployment Trade-Offs & Practical Engineering Recommendation

### Deployment Trade-Offs & Dataset Limitations
- **Missing Network/Host Identifiers (`0 / 40` IP/Port/Host Columns)**: Tabular CICIoT2023 exports strip attacker IPs and concurrent host counts; flow-level models cannot separate `DoS` from `DDoS` or `DDOS-SYNONYMOUSIP_FLOOD` without perimeter packet-header/Flow-key telemetry.
- **Same-Testbed Cross-Capture Shift**: `cross_capture_test_clean.csv` evaluates temporal/file generalization within the same CICIoT2023 testbed (`Merged26–40`) with `<=2,000`/class capping (`5.67×` minority prevalence inflation), so `Cross` rewards high-recall/low-precision priors (`E1`) differently from natural prevalence `Internal Test`.
- **Minority Attack Feature Overlap**: All 39-feature tabular classifiers plateau near `~0.22` Minority Macro-F1 on Test (`~0.35` max recall in `E1` at `0.16` F1 precision collapse), reflecting indistinguishable L7 payload behavior (`SQLINJECTION`, `XSS`, `UPLOADING_ATTACK`) vs. `BENIGN` and `RECON-OSSCAN`.
- **Benign FPR vs. Attack Miss Trade-Off**: Operating at calibrated `sqrt_balanced` or Stage 1 `tau_att=0.15` prevents `7,990` attack misses seen at `T=0.50`, at the cost of `~21–26%` Benign FPR on individual raw flows unless stateful alert aggregation or allow-listing is applied downstream.
- **Merged-Flood Operational Scope Constraint**: `OP29_MERGED_FLOOD` is only valid when downstream SOAR/firewall rules share remediation across DDoS/DoS protocol twins; downstream multi-host source attribution still requires NetFlow/sFlow src-IP tracking outside the ML feature vector.

### Final Engineering Recommendation
- **Primary Production Deployment -> [`E4_LGBM_Leaves63_Sqrt`](file:///c:/Users/priya/Downloads/capstone/track2_ids_engineering/hierarchical_ids/models/e4_lgbm_leaves63_sqrt.joblib)**: Highest Test Accuracy (`75.37%`), highest MCC (`0.7440`), highest ROC-AUC (`0.9823`), highest Cross Accuracy (`70.72%`), smallest single-pass latency (`4.15 ms` P50), single model call (`1.0` exec/flow), and `33.90 MB` footprint. Integrated as the default engine in [`demo_app.py`](file:///c:/Users/priya/Downloads/capstone/track2_ids_engineering/scripts/demo_app.py).
- **Comparative Hierarchical Architecture -> [`E2_Cascade_ValTuned_LGBM`](file:///c:/Users/priya/Downloads/capstone/track2_ids_engineering/hierarchical_ids/results/e2_fast_lgbm_config.json)**: Swapping `binary_lightgbm.joblib` into Stage 1 fixes E2's latency bug (`53.30 ms -> 8.44 ms`, **6.31×** speedup) and matches E4's Macro-F1 (`0.6357` vs. `0.6352`)—recommended only if independent per-stage retraining/thresholds for Flood vs. Non-Flood branches are operationally required.
- **Disqualified Strategy -> `E3_Parallel_*`**: Conditional specialist score fusion without a raw-feature Stage 2 router fails OOD gating (`0.4389` NaiveMax, `0.6239` MetaGate) and increases compute (`2.82` execs) without accuracy gains.

---

## 9. Exact Reproduction & SOC Console Commands

All experiments were executed sequentially under Python 3.13.15 (`numpy 2.5.3`, `pandas 3.0.5`, `scikit-learn 1.9.1`, `lightgbm 4.7.0`, `xgboost 3.4.1`) from repository root `C:\Users\priya\Downloads\capstone`:

```powershell
# 1. Verify existing Track 1 partitions remain untouched & disjoint
.\.venv\Scripts\python.exe -u track1_dataset_evaluation\src\verify_partitions.py

# 2. Train/cache components and run Stage 1-3 + E2/E3 + E4 validation tuning (Train + Val ONLY)
.\.venv\Scripts\python.exe -u -m track2_ids_engineering.hierarchical_ids.experiments.train_and_tune_val --phase components
.\.venv\Scripts\python.exe -u -m track2_ids_engineering.hierarchical_ids.experiments.train_and_tune_val --phase e2_e3
.\.venv\Scripts\python.exe -u -m track2_ids_engineering.hierarchical_ids.experiments.train_and_tune_val --phase e4

# 3. Run frozen read-only evaluation on Val, Internal Test, and Cross-Capture Clean (E1-E4 baseline artifacts)
.\.venv\Scripts\python.exe -u -m track2_ids_engineering.hierarchical_ids.experiments.run_final_evaluation

# 4. Run Phase 1 cleanup additions: Fast E2 (LGBM Stage 1), Merged-Flood Operational E4 Audit, & Figures
.\.venv\Scripts\python.exe -u -m track2_ids_engineering.hierarchical_ids.experiments.run_presentation_updates

# 5. Launch interactive SOC Defense Console (E4 Primary + E2 Fast Toggle + Track 1 Preprocessor)
.\.venv\Scripts\streamlit.exe run track2_ids_engineering/scripts/demo_app.py
```
