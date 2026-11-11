# Comparative Experimental Evaluation of Hierarchical & Improved Flat 34-Class IDS Architectures (CICIoT2023)

> **Research Track**: Track 2 — Practical IDS Engineering ([`track2_ids_engineering/hierarchical_ids/`](file:///c:/Users/priya/Downloads/capstone/track2_ids_engineering/hierarchical_ids))  
> **Upstream Read-Only Reference**: Track 1 — Dataset Evaluation ([`track1_dataset_evaluation/`](file:///c:/Users/priya/Downloads/capstone/track1_dataset_evaluation))  
> **Strict Isolation Status**: All Track 1 code, splits, checkpoints, and results remain unmodified. Tuning occurred strictly on Validation; Internal Test and Cross-Capture Clean were evaluated once read-only under frozen configuration [`frozen_architectures_config.json`](file:///c:/Users/priya/Downloads/capstone/track2_ids_engineering/hierarchical_ids/results/frozen_architectures_config.json).

---

## 1. Executive Summary

This study tests whether hierarchical Flood vs. Non-Flood specialization improves 34-class IoT intrusion detection—especially minority-class detection—over both an existing Track 1 flat baseline (**E1**) and an equally parameterized, class-aware flat classifier (**E4**). We evaluate four architectural families across Track 1's leakage-free Strategy 1 partitions (`Train = 350,821`, `Validation = 75,172`, `Internal Test = 75,105`) and `Cross-Capture Clean = 57,280`:

| Architecture Family | Model Status | Internal Test Acc | Internal Test Macro-F1 | Internal Test MCC | Test Minority F1 (Rec) | Cross Acc | Cross Macro-F1 | Cross MCC | Cross Minority F1 (Rec) | Single-Flow P50 (Test) | Model Execs / Flow | Disk Size |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E1: `E1_Flat34_LGBM_Balanced`** | Reused Track 1 | 72.65% | 0.6145 | 0.7170 | 0.1624 (**0.3360**) | 69.64% | **0.6498** | 0.6861 | **0.3142** (**0.3495**) | **3.43 ms** | 1.00 | **11.33 MB** |
| **E1-Ref: `E1_Ref_Unweighted_Track1`** | Reused Track 1 | 74.45% | 0.6220 | 0.7346 | 0.2014 (0.1531) | 69.13% | 0.6217 | 0.6814 | 0.2343 (0.1580) | 3.63 ms | 1.00 | 11.33 MB |
| **E2 Default: `E2_Cascade_Default_T050`** | Reused S1 + New S2/3 | 73.59% | 0.6185 | 0.7289 | 0.2115 (0.1519) | 68.61% | 0.6213 | 0.6818 | 0.2294 (0.1489) | 7.88 ms | 2.63 | 35.57 MB |
| **E2 Tuned: `E2_Cascade_ValTuned`** | Reused S1 + New S2/3 | 75.04% | **0.6361** | 0.7405 | **0.2264** (0.2094) | 70.50% | 0.6453 | 0.6949 | 0.2954 (0.2214) | 53.30 ms* | 2.82 | 47.00 MB |
| **E3 Naive: `E3_Parallel_NaiveMax`** | Reused S1 + New 3A/3B | 56.11% | 0.4389 | 0.5567 | 0.1757 (0.1084) | 53.26% | 0.4360 | 0.5261 | 0.1707 (0.1021) | 49.73 ms* | 2.82 | 44.65 MB |
| **E3 Temp+Alpha: `E3_Parallel_TempAlpha`** | Reused S1 + New 3A/3B | 54.53% | 0.4301 | 0.5402 | 0.1883 (0.1208) | 51.86% | 0.4273 | 0.5108 | 0.1897 (0.1163) | 49.71 ms* | 2.82 | 44.65 MB |
| **E3 MetaGate: `E3_Parallel_MetaGate`** | Reused S1 + New 3A/3B/Gate | 73.78% | 0.6239 | 0.7275 | 0.2253 (0.2094) | 69.41% | 0.6327 | 0.6837 | 0.2948 (0.2210) | 52.40 ms* | 2.82 | 44.65 MB |
| **E4 Improved Flat: `E4_LGBM_Leaves63_Sqrt`** | Newly trained Track 2 | **75.37%** | 0.6352 | **0.7440** | 0.2196 (0.2063) | **70.72%** | 0.6441 | **0.6972** | 0.2872 (0.2184) | 4.15 ms | 1.00 | 33.90 MB |

*\*Note: `E2_Cascade_ValTuned` and all E3 variants use Track 1's serialized [`binary_random_forest.joblib`](file:///c:/Users/priya/Downloads/capstone/track1_dataset_evaluation/models/binary_random_forest.joblib) (`n_estimators=50, max_depth=15, n_jobs=-1`) in Stage 1; `n_jobs=-1` thread-dispatch overhead adds ~45 ms per single-row query despite `8,042.5 flows/s` batch throughput. By contrast, `E2_bin_lgbm_cat7_sqrt` at `tau_att=0.15` achieved 0.6381 Val Macro-F1 (within 0.0013 of RF Stage 1) at single-flow LGBM latency.*

### Key Scientific Conclusions
1. **Hierarchical specialization improves substantially over the weaker Track 1 baseline (E1), but almost all of that gain comes from fixing Stage 1 thresholding, class weighting (`sqrt_balanced`), and tree capacity (`num_leaves=63, n_estimators=150`) rather than from hierarchy itself.** Once a single flat LightGBM (**E4**) matches those exact capacity and `sqrt_balanced` choices, **E4 essentially ties E2 on Macro-F1** (`0.6352` vs. `0.6361` on Test; `0.6441` vs. `0.6453` on Cross) while **beating E2 on overall Accuracy, MCC, Benign Recall/F1, model disk footprint, execution count (1.0 vs. 2.82), and single-flow latency (4.15 ms vs. 53.30 ms)**.
2. **Stage 2 Flood-vs-Non-Flood routing is near-perfect (`99.98%` Flood Recall, only `50 / 75,105` misroutes on Test), yet flat E4 already separates Flood vs. Non-Flood with fewer cross-family errors (`46` vs. `50`).** Consequently, Stage 2 error propagation is *not* what limits hierarchical performance—nearly 100% of remaining errors occur *inside* Flood (`DDoS` vs. `DoS`, `SYN` vs. `SynonymousIP`) and *inside* Non-Flood/Benign (`RECON-OSSCAN` vs. `BENIGN`/`PORTSCAN`).
3. **Parallel specialist fusion without a raw-feature router (`E3_Parallel_NaiveMax`) collapses (`0.4389` Macro-F1)** because closed-world Flood and Non-Flood specialists both output confidence `>= 0.50` simultaneously on **76.16%** (`52,083`) of passed Test attack flows, yielding `26,678` cross-family errors. A learned Logistic Regression meta-gate (`E3_Parallel_MetaGate`) eliminates 96% of those collisions (`0.6239` Test Macro-F1), yet still underperforms Stage 2 (`1,053` vs. `50` family errors) because meta-features over specialist outputs cannot replace raw packet/flow features.
4. **Minority-class Recall vs. F1 exhibits a sharp weighting and prior-shift trade-off.** Inverse-frequency `balanced` weighting in Track 1 **E1** achieves the highest Minority Macro-Recall (`0.3360` Test; `0.3495` Cross), but extreme rare-class over-prediction crushes its precision on natural Test prevalence (`0.1624` Minority F1, e.g., `651` `VULNERABILITYSCAN` flows mislabeled `SQLINJECTION`). On Cross-Capture Clean—where minority classes were sampled to **5.67x** natural Test prevalence (`14.64%` vs. `2.58%`)—that exact aggressive prior yields higher precision and allows **E1** to reclaim the top Cross Macro-F1 (`0.6498`) despite losing overall Cross Accuracy (`69.64%` vs. `70.72%` E4).

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
| `E2_Cascade_ValTuned` | 0.74914 | **0.63944** | 0.74362 | 0.73924 | **0.79621** (**0.80056**) | **0.48832** (0.48070) | **0.24264** (0.22528) | 0.74203 | 0.25797 | 0.03325 |
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
| `E2_Cascade_ValTuned` | 0.75035 | **0.63614** | 0.74523 | 0.74051 | 0.66220 | 0.97578 | 0.79708 | **0.48104** | **0.22641** (0.20936) | 0.73866 (0.26134) | 0.03294 | 53,297.8 | 64,204.1 | 8,042.5 | 46.996 | 585.8 |
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
| `E2_Cascade_ValTuned` | 0.70503 | 0.64531 | 0.69659 | 0.69493 | 0.68270 | 0.96866 | 0.80863 | 0.49066 | 0.29537 (0.22137) | 0.74528 (0.25472) | 0.04849 | 57,609.6 | 73,203.7 | 7,958.2 |
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

---

## 7. Stage-by-Stage Error Attribution & Diagnostic Analysis

### 7.1 Full Error Decomposition Across Splits ([`error_attribution_test.csv`](file:///c:/Users/priya/Downloads/capstone/track2_ids_engineering/hierarchical_ids/results/error_attribution_test.csv), [`error_attribution_cross.csv`](file:///c:/Users/priya/Downloads/capstone/track2_ids_engineering/hierarchical_ids/results/error_attribution_cross.csv))
Every misclassified flow falls into one of six mutually exclusive buckets:

| Split & Architecture | Total Errors | Attack->Benign (FN) | Benign->Attack (FP) | Flood -> Non-Flood | Non-Flood -> Flood | Within-Flood Confusion | Within-NonFlood Confusion |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Test** `E1_Flat34_LGBM_Balanced` | 20,539 | **1,517** | 2,064 | 16 | 34 | 8,134 | 8,774 |
| **Test** `E2_Cascade_Default_T050` | 19,836 | 7,990 | **158** | 21 | 29 | **8,052** | **3,586** *(masked by FN)* |
| **Test** `E2_Cascade_ValTuned` | 18,750 | 2,276 | 1,573 | **9** | 41 | 8,055 | 6,796 |
| **Test** `E3_Parallel_NaiveMax` | 32,964 | 2,276 | 1,573 | 13,347 | 13,331 | 1,866 | 571 |
| **Test** `E3_Parallel_MetaGate` | 19,696 | 2,276 | 1,573 | 966 | 87 | 8,005 | 6,789 |
| **Test** `E4_LGBM_Leaves63_Sqrt` | **18,501** | 2,379 | 1,286 | 11 | **35** | 8,087 | 6,703 |
| **Cross** `E1_Flat34_LGBM_Balanced` | 17,390 | **1,507** | 1,718 | 19 | 15 | 4,388 | 9,743 |
| **Cross** `E2_Cascade_Default_T050` | 17,983 | 9,329 | **134** | 30 | **14** | 4,367 | 4,109 |
| **Cross** `E2_Cascade_ValTuned` | 16,896 | 2,536 | 1,269 | **11** | 19 | 4,377 | 8,684 |
| **Cross** `E3_Parallel_NaiveMax` | 26,773 | 2,536 | 1,269 | 7,473 | 13,813 | 1,138 | 544 |
| **Cross** `E3_Parallel_MetaGate` | 17,520 | 2,536 | 1,269 | 653 | 44 | 4,340 | 8,678 |
| **Cross** `E4_LGBM_Leaves63_Sqrt` | **16,771** | 2,646 | **1,080** | **11** | 16 | 4,373 | **8,645** |

### 7.2 Answers to Core Architectural Questions (Experimental Evidence)

1. **Why did Stage 1 fail so heavily at threshold `0.50` (`E2_Cascade_Default_T050`)?**  
   Track 1 trained `binary_lightgbm.joblib` with `class_weight="balanced"`. Because Attacks outnumber Benign `11.6:1` in Train (`322,966 : 27,855`), each Benign sample received `~11.6×` weight, making the default `0.50` boundary overly conservative toward Benign: `97.38%` Test Benign Recall came at the cost of exiting **`7,990` (11.57%)** true Test attacks and **`9,329` (17.84%)** true Cross attacks at Stage 1 before Stage 2 ever saw them. Tuning `tau_att = 0.15` reduced missed attacks to `2,276` (Test) and `2,536` (Cross), raising Test Macro-F1 by `+0.0177` (`0.6185 -> 0.6361`) and Cross Macro-F1 by `+0.0240` (`0.6213 -> 0.6453`), while raising Benign FPR (`2.63% -> 26.13%` Test).
2. **Is Stage 2 routing a bottleneck in E2?**  
   **No.** On Test (`stage_diagnostics_test.json`), `stage2_cat7_lgbm_sqrt_balanced` achieved `87.08%` 7-category accuracy on passed attacks and **`99.98%` Flood Recall / `99.90%` Flood Precision**, committing only **50 total Flood/Non-Flood misroutes out of 75,105 flows** (`9` Flood->NonFlood + `41` NonFlood->Flood) on Test, and only **30 out of 57,280** (`11 + 19`) on Cross. Training Stage 2 strictly on the 7 attack categories eliminated all `2,393` invalid Stage 2 Benign exits observed with Track 1's 8-class router.
3. **Which traffic family benefited from specialization (`E2` vs. `E1`), and did `E4` close the gap?**  
   - On Internal Test (`E2` vs. `E1`): **Non-Flood** Macro-F1 rose from `0.4410` to **`0.4810`** (`+0.0400`), whereas **Flood** rose only `0.7943` -> `0.7971` (`+0.0028`).
   - Comparing **`E2_Cascade_ValTuned` vs. capacity/weight-matched `E4_LGBM_Leaves63_Sqrt`**: E4 already matches Flood (`0.7960` Test, `0.8089` Cross) and virtually ties Non-Flood (`0.4789` Test, `0.4870` Cross), winning **16 of 34 classes** by F1 on Test (`16` E2 wins, `2` ties) and trailing `16` to `17` (`1` tie) on Cross. E2's slight Macro-F1 edge (`+0.0009` Test, `+0.0012` Cross) comes mostly from `RECON-PINGSWEEP` (`+0.0674` F1 Test / `+0.0419` Cross), `SQLINJECTION` (`+0.0214` Test), and `XSS` (`+0.0248` Cross), while E4 beats E2 on `BENIGN` (`+0.0230` Test / `+0.0156` Cross), `DICTIONARYBRUTEFORCE`, `VULNERABILITYSCAN`, and `UPLOADING_ATTACK`.
   - Meanwhile, on **pairwise Test flips** ([`pairwise_error_analysis_test.json`](file:///c:/Users/priya/Downloads/capstone/track2_ids_engineering/hierarchical_ids/results/pairwise_error_analysis_test.json)): `E2` fixed `3,521` E1 errors and broke `1,732` (`+1,789` net); `E4` fixed `3,472` E1 errors and broke `1,434` (`+2,038` net).
4. **Why did `E3_Parallel_NaiveMax` & `TempAlpha` collapse, and how well did `MetaGate` recover?**  
   - Conditionally trained specialists (`Stage 3A` on Floods, `Stage 3B` on Non-Floods) never learn to reject out-of-distribution inputs from the opposite family. Consequently, **`76.16%` (`52,083` Test)** and **`71.22%` (`36,346` Cross)** of passed attacks triggered **`>= 0.50` probability simultaneously in both 3A and 3B**! Direct score comparison caused **`26,678` cross-family errors** on Test (`43.89%` Macro-F1).
   - Static scalar temperature scaling + `alpha=0.50` did not fix OOD overconfidence (`0.4301` Macro-F1). Adding **`E3_Parallel_MetaGate`** cut family errors by 96% (`26,678 -> 1,053` Test; `21,286 -> 697` Cross; `84.16%` accuracy when both specialists are confident), recovering Macro-F1 to `0.6239` Test / `0.6327` Cross. However, meta-features over closed-world predictions still underperformed Stage 2 (`966` Flood->NonFlood leakage, notably dropping `DOS-UDP_FLOOD` F1 from `0.6778` to `0.4573` on Test and `0.7216` to `0.4809` on Cross). Against E2's `50` Test Stage 2 misroutes, E3 MetaGate recovered only `17` while breaking `963` correct E2 predictions.
5. **What causes the persistent error ceiling across all architectures (`~75%` Test Acc / `~0.64` Macro-F1)?**  
   Inspecting the top 34×34 confusion pairs shows **every top architecture (E1, E2, E4) collides on the exact same ambiguous packet/flow pairs**:
   - **Within-Flood DoS vs. DDoS & SYN pairs (Test `E2` / `E4`)**: `DDOS-SYN_FLOOD -> DDOS-SYNONYMOUSIP_FLOOD` (`1,281` / `1,295`), `DDOS-TCP_FLOOD -> DOS-TCP_FLOOD` (`1,162` / `1,182`), `DOS-UDP_FLOOD -> DDOS-UDP_FLOOD` (`1,090` / `1,093`), `DOS-SYN_FLOOD -> DDOS-SYNONYMOUSIP_FLOOD` (`995` / `1,001`). Single-flow features cannot easily count concurrent distinct attacker IPs (DDoS vs. DoS).
   - **Low-volume Non-Flood vs. Benign**: `RECON-OSSCAN -> BENIGN` (`758` E2 / `820` E4), `RECON-OSSCAN -> RECON-PORTSCAN` (`711` / `712`), `RECON-PORTSCAN -> VULNERABILITYSCAN` (`621` / `612`), and stealthy attack misses (`RECON-PORTSCAN`, `VULNERABILITYSCAN`, `MITM-ARPSPOOFING`, `DICTIONARYBRUTEFORCE -> BENIGN`).

---

## 8. Limitations & Practical Engineering Recommendation

### Key Limitations
- **Same-Testbed Holdout Shift**: `cross_capture_test_clean.csv` validates temporal/capture-file generalization *within* the CICIoT2023 lab rather than cross-network deployment, and its ≤2,000/class capping alters minority class priors by `5.67×`.
- **Stage 1 RF Single-Flow Threadpool Overhead**: Reusing Track 1's unmodified `binary_random_forest.joblib` (`n_jobs=-1`) yields strong batch throughput (`8,042.5 fps`) but high per-row latency (`53.3 ms` P50) due to per-call worker synchronization.
- **Residual Minority Recall Ceiling**: No tested 39-feature tabular classifier exceeds `0.23` Minority Macro-F1 on Test (`0.35` Max Minority Recall with `E1`'s `0.16` F1 precision collapse) due to overlapping single-flow representations across Web/Recon attacks and Benign traffic.

### Engineering Recommendation
- **For Production / Low-Latency Single-Flow Deployment -> Choose [`E4_LGBM_Leaves63_Sqrt`](file:///c:/Users/priya/Downloads/capstone/track2_ids_engineering/hierarchical_ids/models/e4_lgbm_leaves63_sqrt.joblib)** (`1.0` execution/flow, `4.15 ms` Test P50 latency, `33.90 MB`, **highest Accuracy `75.37%` and MCC `0.7440`**, highest Benign Recall among non-default models `78.63%`, virtually tied Macro-F1 `0.6352` vs. `0.6361`).
- **For Modular Policy Separation (Tuning Flood vs. Non-Flood Weights Independently) -> Choose `E2_Cascade_ValTuned`** (or its LightGBM Stage 1 variant `E2_bin_lgbm_cat7_sqrt` at `tau_att=0.15`, which keeps Val Macro-F1 `0.6381` while eliminating RF thread overhead).
- **Avoid Naive Parallel Fusion (`E3_NaiveMax` / `E3_TempAlpha`)** unless paired with a raw-feature router; conditional specialist outputs are not comparable across disjoint class manifolds without a calibration gate or explicit Stage 2 router.

---

## 9. Exact Reproduction Commands

All experiments were executed sequentially under Python 3.13.15 (`numpy 2.5.3`, `pandas 3.0.5`, `scikit-learn 1.9.1`, `lightgbm 4.7.0`, `xgboost 3.4.1`) from repository root `C:\Users\priya\Downloads\capstone`:

```powershell
# 1. Verify existing Track 1 partitions remain untouched & disjoint
.\.venv\Scripts\python.exe -u track1_dataset_evaluation\src\verify_partitions.py

# 2. Train/cache components and run Stage 1-3 + E2/E3 + E4 validation tuning (Train + Val ONLY)
.\.venv\Scripts\python.exe -u -m track2_ids_engineering.hierarchical_ids.experiments.train_and_tune_val --phase components
.\.venv\Scripts\python.exe -u -m track2_ids_engineering.hierarchical_ids.experiments.train_and_tune_val --phase e2_e3
.\.venv\Scripts\python.exe -u -m track2_ids_engineering.hierarchical_ids.experiments.train_and_tune_val --phase e4

# 3. Run frozen read-only evaluation on Val, Internal Test, and Cross-Capture Clean
.\.venv\Scripts\python.exe -u -m track2_ids_engineering.hierarchical_ids.experiments.run_final_evaluation
```
