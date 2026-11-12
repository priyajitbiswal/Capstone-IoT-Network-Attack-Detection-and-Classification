# Phase 1: Dataset, Partition & Baseline Inspection Report and Experiment Plan

**Scope:** Controlled minority-class optimization experiments inside [`track2_ids_engineering/minority_optimization/`](file:///C:/Users/priya/Downloads/capstone/track2_ids_engineering/minority_optimization) under strict zero-leakage and artifact-preservation constraints (Track 1, existing E1–E4 checkpoints, partitions, and historical results remain untouched).

---

## 1. Dataset, Duplicate/Collision Audit & Partition Verification

Verified directly against `data/sample_stratified.csv`, [`partition_metadata.json`](file:///C:/Users/priya/Downloads/capstone/track1_dataset_evaluation/data/partitions/partition_metadata.json), and [`cross_capture_metadata.json`](file:///C:/Users/priya/Downloads/capstone/track1_dataset_evaluation/data/partitions/cross_capture_metadata.json):

| Quantity | Verified Count | Notes |
| --- | ---: | --- |
| Original CICIoT2023 stratified sample (`data/sample_stratified.csv`) | `529,562` | Exact source rows |
| Exact `(features + label)` duplicates removed | `28,464` | `5.375%` redundant rows (`47,430` rows in exact duplicate clusters) |
| Clean deduplicated dataset | `501,098` | Split into Train / Val / Internal Test |
| Unique feature vectors in clean dataset | `491,422` | Zero feature-hash overlap across Train / Val / Internal Test |
| Conflicting-label feature-hash collision groups | `8,740` | `28,608` rows (`5.402%`) where identical 39-feature vectors carry >1 label |
| **Training split** (`train`) | **`350,821`** | `70.01%`; only split ever seen during fitting, SMOTE, or weight computation |
| **Validation split** (`val`) | **`75,172`** | `15.00%`; used solely for hyperparameter/threshold/policy selection |
| **Internal Test split** (`test`) | **`75,105`** | `14.99%`; held-out read-only evaluation |
| **Cross-Capture Test split** (`cross_capture_test_clean.csv`) | **`57,280`** | Deduplicated from `63,423` extracted cross-capture rows; zero exact hash overlap |

### Exact 39 Input Features (`FEATURE_COLUMNS`)

```text
 1. Header_Length       14. ack_count     27. ICMP       39. Variance
 2. Protocol Type       15. syn_count     28. IGMP
 3. Time_To_Live        16. fin_count     29. IPv
 4. Rate                17. rst_count     30. LLC
 5. fin_flag_number     18. HTTP          31. Tot sum
 6. syn_flag_number     19. HTTPS         32. Min
 7. rst_flag_number     20. DNS           33. Max
 8. psh_flag_number     21. Telnet        34. AVG
 9. ack_flag_number     22. SMTP          35. Std
10. ece_flag_number     23. SSH           36. Tot size
11. cwr_flag_number     24. IRC           37. IAT
12. UDP (col 24 idx)    25. TCP           38. Number
13. DHCP (col 25 idx)   26. ARP
```

Explicit ordered list (`0..38`) from [`experiment_plan.json`](file:///C:/Users/priya/Downloads/capstone/track2_ids_engineering/minority_optimization/configs/experiment_plan.json):
`Header_Length`, `Protocol Type`, `Time_To_Live`, `Rate`, `fin_flag_number`, `syn_flag_number`, `rst_flag_number`, `psh_flag_number`, `ack_flag_number`, `ece_flag_number`, `cwr_flag_number`, `ack_count`, `syn_count`, `fin_count`, `rst_count`, `HTTP`, `HTTPS`, `DNS`, `Telnet`, `SMTP`, `SSH`, `IRC`, `TCP`, `UDP`, `DHCP`, `ARP`, `ICMP`, `IGMP`, `IPv`, `LLC`, `Tot sum`, `Min`, `Max`, `AVG`, `Std`, `Tot size`, `IAT`, `Number`, `Variance`.

### Exact 8 Minority Classes (`MINORITY_ATTACK_CLASSES`)

Used verbatim by the repository's `Minority F1` and `Minority Recall` macro-averages (all 8 belong to the **Non-Flood** operational family and remain unmerged under **OP29**):

1. `UPLOADING_ATTACK` (`Train=312`, `Val=44`, `Test=60`, `Cross=289`)
2. `RECON-PINGSWEEP` (`Train=486`, `Val=111`, `Test=125`, `Cross=567`)
3. `BACKDOOR_MALWARE` (`Train=766`, `Val=155`, `Test=139`, `Cross=729`)
4. `XSS` (`Train=887`, `Val=169`, `Test=183`, `Cross=925`)
5. `COMMANDINJECTION` (`Train=1,186`, `Val=225`, `Test=255`, `Cross=1,322`)
6. `SQLINJECTION` (`Train=1,212`, `Val=252`, `Test=254`, `Cross=1,233`)
7. `BROWSERHIJACKING` (`Train=1,324`, `Val=271`, `Test=294`, `Cross=1,320`)
8. `DICTIONARYBRUTEFORCE` (`Train=2,935`, `Val=617`, `Test=629`, `Cross=1,999`)

- **Combined Rare Minority Pool:** `Train=9,108` (`2.60%`), `Val=1,844` (`2.45%`), `Test=1,939` (`2.58%`), `Cross=8,384` (`14.64%`).
- **Zero-Flood Specialist Training Pool:** `161,533` rows (`9,108` rare minority positives + `27,855` `BENIGN` + `124,570` common Non-Flood rows; **`0` Flood rows**).

---

## 2. Exact Class Distributions

### 2A. Original 34-Class Taxonomy (`ORIG34`)

Source: [`phase1_class_distribution_34.csv`](file:///C:/Users/priya/Downloads/capstone/track2_ids_engineering/minority_optimization/results/phase1_class_distribution_34.csv)

| Class (`34`) | Category | OP29 Target | Family | Minority8 | Raw Sample | Clean Total | Train | Val | Test | Cross-Capture |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `BACKDOOR_MALWARE` | Web | `BACKDOOR_MALWARE` | Non-Flood | **Yes** | 1,061 | 1,060 | 766 | 155 | 139 | 729 |
| `BENIGN` | Benign | `BENIGN` | Benign | No | 40,000 | 39,960 | 27,855 | 6,086 | 6,019 | 4,982 |
| `BROWSERHIJACKING` | Web | `BROWSERHIJACKING` | Non-Flood | **Yes** | 1,904 | 1,889 | 1,324 | 271 | 294 | 1,320 |
| `COMMANDINJECTION` | Web | `COMMANDINJECTION` | Non-Flood | **Yes** | 1,666 | 1,666 | 1,186 | 225 | 255 | 1,322 |
| `DDOS-ACK_FRAGMENTATION` | DDoS | `DDOS-ACK_FRAGMENTATION` | Flood | No | 20,000 | 19,989 | 13,968 | 3,053 | 2,968 | 2,000 |
| `DDOS-HTTP_FLOOD` | DDoS | `FLOOD_HTTP` | Flood | No | 9,121 | 9,121 | 6,395 | 1,424 | 1,302 | 2,000 |
| `DDOS-ICMP_FLOOD` | DDoS | `DDOS-ICMP_FLOOD` | Flood | No | 20,000 | 16,628 | 11,639 | 2,485 | 2,504 | 1,428 |
| `DDOS-ICMP_FRAGMENTATION` | DDoS | `DDOS-ICMP_FRAGMENTATION` | Flood | No | 20,000 | 19,993 | 14,029 | 2,947 | 3,017 | 2,000 |
| `DDOS-PSHACK_FLOOD` | DDoS | `DDOS-PSHACK_FLOOD` | Flood | No | 20,000 | 17,714 | 12,435 | 2,598 | 2,681 | 1,592 |
| `DDOS-RSTFINFLOOD` | DDoS | `DDOS-RSTFINFLOOD` | Flood | No | 20,000 | 17,101 | 11,804 | 2,642 | 2,655 | 1,463 |
| `DDOS-SLOWLORIS` | DDoS | `DDOS-SLOWLORIS` | Flood | No | 7,534 | 7,534 | 5,273 | 1,102 | 1,159 | 2,000 |
| `DDOS-SYNONYMOUSIP_FLOOD` | DDoS | `FLOOD_SYN` | Flood | No | 20,000 | 17,253 | 12,184 | 2,610 | 2,459 | 1,273 |
| `DDOS-SYN_FLOOD` | DDoS | `FLOOD_SYN` | Flood | No | 20,000 | 17,690 | 12,374 | 2,686 | 2,630 | 1,368 |
| `DDOS-TCP_FLOOD` | DDoS | `FLOOD_TCP` | Flood | No | 20,000 | 16,539 | 11,585 | 2,503 | 2,451 | 1,273 |
| `DDOS-UDP_FLOOD` | DDoS | `FLOOD_UDP` | Flood | No | 20,000 | 17,375 | 12,148 | 2,593 | 2,634 | 1,442 |
| `DDOS-UDP_FRAGMENTATION` | DDoS | `DDOS-UDP_FRAGMENTATION` | Flood | No | 20,000 | 19,991 | 13,956 | 3,031 | 3,004 | 2,000 |
| `DICTIONARYBRUTEFORCE` | BruteForce | `DICTIONARYBRUTEFORCE` | Non-Flood | **Yes** | 4,181 | 4,181 | 2,935 | 617 | 629 | 1,999 |
| `DNS_SPOOFING` | Spoofing | `DNS_SPOOFING` | Non-Flood | No | 20,000 | 19,675 | 13,733 | 2,905 | 3,037 | 1,948 |
| `DOS-HTTP_FLOOD` | DoS | `FLOOD_HTTP` | Flood | No | 20,000 | 19,955 | 14,040 | 2,973 | 2,942 | 1,996 |
| `DOS-SYN_FLOOD` | DoS | `FLOOD_SYN` | Flood | No | 20,000 | 18,072 | 12,579 | 2,738 | 2,755 | 1,447 |
| `DOS-TCP_FLOOD` | DoS | `FLOOD_TCP` | Flood | No | 20,000 | 17,249 | 12,025 | 2,597 | 2,627 | 1,374 |
| `DOS-UDP_FLOOD` | DoS | `FLOOD_UDP` | Flood | No | 20,000 | 18,355 | 12,854 | 2,778 | 2,723 | 1,591 |
| `MIRAI-GREETH_FLOOD` | Mirai | `MIRAI-GREETH_FLOOD` | Non-Flood | No | 20,000 | 19,942 | 14,072 | 2,912 | 2,958 | 1,988 |
| `MIRAI-GREIP_FLOOD` | Mirai | `MIRAI-GREIP_FLOOD` | Non-Flood | No | 20,000 | 19,965 | 13,887 | 2,992 | 3,086 | 1,994 |
| `MIRAI-UDPPLAIN` | Mirai | `MIRAI-UDPPLAIN` | Non-Flood | No | 20,000 | 19,912 | 14,004 | 2,959 | 2,949 | 1,985 |
| `MITM-ARPSPOOFING` | Spoofing | `MITM-ARPSPOOFING` | Non-Flood | No | 20,000 | 18,721 | 13,206 | 2,752 | 2,763 | 1,847 |
| `RECON-HOSTDISCOVERY` | Recon | `RECON-HOSTDISCOVERY` | Non-Flood | No | 20,000 | 19,930 | 14,010 | 2,989 | 2,931 | 1,983 |
| `RECON-OSSCAN` | Recon | `RECON-OSSCAN` | Non-Flood | No | 20,000 | 19,815 | 13,895 | 2,952 | 2,968 | 1,969 |
| `RECON-PINGSWEEP` | Recon | `RECON-PINGSWEEP` | Non-Flood | **Yes** | 722 | 722 | 486 | 111 | 125 | 567 |
| `RECON-PORTSCAN` | Recon | `RECON-PORTSCAN` | Non-Flood | No | 20,000 | 19,749 | 13,790 | 3,009 | 2,950 | 1,959 |
| `SQLINJECTION` | Web | `SQLINJECTION` | Non-Flood | **Yes** | 1,718 | 1,718 | 1,212 | 252 | 254 | 1,233 |
| `UPLOADING_ATTACK` | Web | `UPLOADING_ATTACK` | Non-Flood | **Yes** | 416 | 416 | 312 | 44 | 60 | 289 |
| `VULNERABILITYSCAN` | Recon | `VULNERABILITYSCAN` | Non-Flood | No | 20,000 | 19,979 | 13,973 | 3,012 | 2,994 | 1,994 |
| `XSS` | Web | `XSS` | Non-Flood | **Yes** | 1,239 | 1,239 | 887 | 169 | 183 | 925 |
| **Total (34 classes)** | — | **29 classes** | — | **8 classes** | **529,562** | **501,098** | **350,821** | **75,172** | **75,105** | **57,280** |

### 2B. Merged OP29 Flood-Twin Groups (`OP29`)

Source: [`phase1_class_distribution_op29.csv`](file:///C:/Users/priya/Downloads/capstone/track2_ids_engineering/minority_optimization/results/phase1_class_distribution_op29.csv) (only the 4 merged Flood superclasses differ from 34-class counts; the remaining 25 classes retain identical counts):

| OP29 Class | Merged From | Raw Sample | Clean Total | Train | Val | Test | Cross-Capture |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `FLOOD_HTTP` | `DDOS-HTTP_FLOOD` \| `DOS-HTTP_FLOOD` | 29,121 | 29,076 | 20,435 | 4,397 | 4,244 | 3,996 |
| `FLOOD_SYN` | `DDOS-SYNONYMOUSIP_FLOOD` \| `DDOS-SYN_FLOOD` \| `DOS-SYN_FLOOD` | 60,000 | 53,015 | 37,137 | 8,034 | 7,844 | 4,088 |
| `FLOOD_TCP` | `DDOS-TCP_FLOOD` \| `DOS-TCP_FLOOD` | 40,000 | 33,788 | 23,610 | 5,100 | 5,078 | 2,647 |
| `FLOOD_UDP` | `DDOS-UDP_FLOOD` \| `DOS-UDP_FLOOD` | 40,000 | 35,730 | 25,002 | 5,371 | 5,357 | 3,033 |

---

## 3. Existing Frozen E4 Baselines (`E4_LGBM_Leaves63_Sqrt`)

Source: [`phase1_baseline_e4.csv`](file:///C:/Users/priya/Downloads/capstone/track2_ids_engineering/minority_optimization/results/phase1_baseline_e4.csv) and [`phase1_baseline_e4_per_class_test.csv`](file:///C:/Users/priya/Downloads/capstone/track2_ids_engineering/minority_optimization/results/phase1_baseline_e4_per_class_test.csv) (verified bit-exact parity with `hierarchical_ids` historical results).

### Summary Metrics Across Splits

| Baseline Model | Taxonomy | Split | Accuracy | Macro-F1 | Weighted-F1 | MCC | Minority F1 | Minority Recall | Benign Recall | Benign FPR | Attack→Benign FNR |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `E4_Baseline_34` | ORIG34 | Val | 0.75133 | 0.63576 | 0.74457 | 0.74157 | 0.22709 | 0.21942 | 0.78426 | 0.21574 | 0.03507 |
| `E4_Baseline_34` | ORIG34 | Test | 0.75366 | 0.63523 | 0.74721 | 0.74402 | 0.21957 | 0.20631 | 0.78634 | 0.21366 | 0.03444 |
| `E4_Baseline_34` | ORIG34 | Cross | 0.70721 | 0.64405 | 0.69703 | 0.69715 | 0.28715 | 0.21835 | 0.78322 | 0.21678 | 0.05059 |
| `E4_Baseline_OP29_Remap` | OP29 | Val | 0.85822 | 0.68244 | 0.85350 | 0.85048 | 0.22709 | 0.21942 | 0.78426 | 0.21574 | 0.03507 |
| `E4_Baseline_OP29_Remap` | OP29 | Test | 0.85805 | 0.68082 | 0.85341 | 0.85038 | 0.21957 | 0.20631 | 0.78634 | 0.21366 | 0.03444 |
| `E4_Baseline_OP29_Remap` | OP29 | Cross | 0.78036 | 0.68527 | 0.77081 | 0.77123 | 0.28715 | 0.21835 | 0.78322 | 0.21678 | 0.05059 |

### E4 Baseline Per-Class Performance for the 8 Minority Classes (Internal Test)

| Minority Class | Train Support | Test Support | E4 Precision | E4 Recall | E4 F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `UPLOADING_ATTACK` | 312 | 60 | 0.04000 | 0.03333 | 0.03636 |
| `XSS` | 887 | 183 | 0.11594 | 0.08743 | 0.09969 |
| `RECON-PINGSWEEP` | 486 | 125 | 0.13636 | 0.09600 | 0.11268 |
| `SQLINJECTION` | 1,212 | 254 | 0.15854 | 0.10236 | 0.12440 |
| `BACKDOOR_MALWARE` | 766 | 139 | 0.13636 | 0.15108 | 0.14334 |
| `COMMANDINJECTION` | 1,186 | 255 | 0.33730 | 0.33333 | 0.33531 |
| `BROWSERHIJACKING` | 1,324 | 294 | 0.48583 | 0.40816 | 0.44362 |
| `DICTIONARYBRUTEFORCE` | 2,935 | 629 | 0.48592 | 0.43879 | 0.46115 |

*(Key non-Flood bottleneck outside the 8 minority classes on Internal Test: `RECON-OSSCAN` F1=`0.27656`, `RECON-PORTSCAN` F1=`0.42174`, `VULNERABILITYSCAN` F1=`0.51193`, `BENIGN` F1=`0.72089` with `21.37%` FPR.)*

---

## 4. Controlled Experiment Plan (`RANDOM_SEED = 42`)

All new experiments live strictly in `track2_ids_engineering/minority_optimization/` and use the frozen `E4_LGBM_PARAMS` base configuration (`n_estimators=150, learning_rate=0.05, num_leaves=63, random_state=42, n_jobs=-1`). No class weighting is stacked onto SMOTE, and no SMOTE is used in Class Weighting, isolating each intervention cleanly.

1. **Experiment 1 — Train-Only SMOTE + Realism Audit (separate `ORIG34` and `OP29`):**
   - Since `imbalanced-learn` is not installed in `.venv` (kept untouched), implement deterministic Chawla (2002) $k=5$ NN linear interpolation on training rows only.
   - Run a pre-training feature-realism audit quantifying: fractional/mixed discrete `Protocol Type` values (`{0, 1, 6, 17, 47}`), violated quadratic identity `Variance == Std^2`, non-integer packet counts, and Train 5-NN neighborhood purity for each minority class.
   - Evaluate 4 controlled candidates per taxonomy (`alpha=0.0` unweighted):
     1. `SMOTE_Std_Cap4000` (unconstrained standard 5-NN SMOTE up to `min(4000, 8*n_c)` on all 8 minority classes)
     2. `SMOTE_ProtoSafe_Cap4000` (within-`Protocol Type` 5-NN + invariant repair, skip sub-strata `< 10` rows)
     3. `SMOTE_ProtoSafe_Cap8000` (within-`Protocol Type` 5-NN + invariant repair, higher `min(8000, 10*n_c)` target)
     4. `SMOTE_ProtoSafe_Selective` (skip classes/seeds with severe class overlap / `<15%` 5-NN purity so synthetic samples are not injected directly inside dense `BENIGN` / scan clouds; cap `6000`)
   - Select the best SMOTE configuration per taxonomy strictly on Validation (`Minority F1` + `Macro-F1` subject to Benign FPR & Accuracy guardrails).

2. **Experiment 2 — Isolated Class Weighting (`alpha in {0.25, 0.50, 0.75, 1.00}`, separate `ORIG34` and `OP29`):**
   - No oversampling (`SMOTE = False`). Normalized inverse-frequency power weights $w_c \propto N_c^{-\alpha}$ scaled so $\frac{1}{|C|}\sum_c w_c = 1$:
     - `Power_a025` ($\alpha = 0.25$)
     - `SqrtBalanced_a050` ($\alpha = 0.50$)
     - `Power_a075` ($\alpha = 0.75$)
     - `Balanced_a100` ($\alpha = 1.00$)
   - Select the Validation-optimal weighting exponent per taxonomy. Note that under OP29, `SqrtBalanced` weights are recomputed over the 29 merged training counts rather than simply relabeling a 34-class model's output.

3. **Experiment 3 — Zero-Flood Rare Non-Flood Specialist + Validation-Gated Hybrid (separate `ORIG34` and `OP29`):**
   - **Zero-Flood Training Pool (`161,533` rows):** Strictly `0` Flood attack rows (`9,108` rare minority positives vs `152,425` `BENIGN` + common Non-Flood negatives; verified by runtime assertion).
   - **Two-Stage Specialist:**
     - Stage A: LightGBM binary Rare-Non-Flood vs Safe/Common-Non-Flood detector (Val-selected over `weight alpha in {0.0, 0.50, 1.00}`).
     - Stage B: LightGBM 8-class classifier over the 8 minority classes (`9,108` Train rows, Val-selected over `weight alpha in {0.0, 0.50, 1.00}`).
   - **Integration with Frozen E4 Base Model (E4-34 or E4-OP29 native):**
     - Flows predicted by E4 as Flood remain untouched (note: on Validation/Test, `err_nonflood_to_flood <= 35` rows vs `~40,500+` true Floods).
     - Flows predicted by E4 as Non-Flood (including `BENIGN`) are evaluated by Stage A + Stage B. Compare a **Naive `tau=0.50` override** against a **Validation-tuned gating policy** `(tau_rare, tau_cls, tau_e4_keep)` optimizing Macro-F1 / Minority F1 without exploding Benign FPR.

4. **Phase 4 & 5 — Read-Only Internal Test & Cross-Capture Evaluation, Plots, and `FINAL_REPORT.md`:**
   - Freeze all Validation-selected models/policies and run unified single-session evaluation across `Val (75,172)`, `Test (75,105)`, and `Cross-Capture (57,280)` recording Accuracy, Macro-F1, Weighted-F1, MCC, Minority F1/Recall, Benign Recall/FPR, Attack→Benign FNR, P50/P95 latency, Batch throughput, Model size, Per-class CSVs, and Confusion Matrices separately for `ORIG34` and `OP29`.
