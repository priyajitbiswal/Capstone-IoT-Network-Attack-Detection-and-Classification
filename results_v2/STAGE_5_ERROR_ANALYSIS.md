# Stage 5: Fine-Grained Error Analysis Report

**Analysis Scope**: Fine-grained error profiling, confusion matrix decomposition, and ablation comparisons for 34-class attack classification on the CICIoT2023 benchmark.  
**Primary Evaluation Artifacts**:
- [`results_v2/metrics/stage4/fine_grained_lightgbm_stage4.json`](metrics/stage4/fine_grained_lightgbm_stage4.json)
- [`results_v2/metrics/stage4/ablation_unweighted_fine_grained_lightgbm_stage4.json`](metrics/stage4/ablation_unweighted_fine_grained_lightgbm_stage4.json)
- [`results_v2/metrics/stage4/ablation_s2_fine_grained_lightgbm_stage4.json`](metrics/stage4/ablation_s2_fine_grained_lightgbm_stage4.json)
- [`results_v2/tables/stage4_test_results.csv`](tables/stage4_test_results.csv)
**Analysis Script Reference**: `scratch/stage5_error_analysis.py`

---

## 1. Top 10 Worst-Performing Attack Classes (`fine_grained_lightgbm`)

The 10 attack classes with the lowest classification performance were identified using the internal test partition ($N = 75,105$), ranked in ascending order by test Macro-F1.

### 1.1 Measured Empirical Results

| Rank | Attack Class | Threat Tier | Test F1 | Test Recall | Test Precision | Support ($N$) | Cross-Capture F1 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | `UPLOADING_ATTACK` | Tier 4 (Web) | **0.0545** | 23.33% | 3.08% | 60 | 0.1135 |
| **2** | `XSS` | Tier 4 (Web) | **0.0715** | 20.22% | 4.34% | 183 | 0.2138 |
| **3** | `RECON-PINGSWEEP` | Tier 3 (Recon) | **0.0754** | 26.40% | 4.40% | 125 | 0.2366 |
| **4** | `SQLINJECTION` | Tier 4 (Web) | **0.0834** | 35.83% | 4.72% | 254 | 0.2846 |
| **5** | `BACKDOOR_MALWARE` | Tier 4 (Web) | **0.0968** | 25.18% | 5.99% | 139 | 0.2197 |
| **6** | `RECON-OSSCAN` | Tier 3 (Recon) | **0.2342** | 16.85% | 38.43% | 2,968 | 0.1991 |
| **7** | `COMMANDINJECTION` | Tier 4 (Web) | **0.2587** | 32.16% | 21.64% | 255 | 0.3975 |
| **8** | `BROWSERHIJACKING` | Tier 4 (Web) | **0.2870** | 53.06% | 19.67% | 294 | 0.5101 |
| **9** | `DDOS-SYN_FLOOD` | Tier 1 (Flood) | **0.3632** | 30.11% | 45.75% | 2,630 | 0.4385 |
| **10** | `DICTIONARYBRUTEFORCE` | Tier 4 (Web) | **0.3719** | 52.62% | 28.76% | 629 | 0.5378 |

### 1.2 Measured Findings vs. Hypothesized Failure Causes
- **Measured Finding**: Seven of the 10 worst-performing classes belong to **Tier 4 (Web & Exploit Attacks)**, with support counts below 650 flows in the internal test split. Two belong to **Tier 3 (Reconnaissance & Spoofing)**, and only one belongs to **Tier 1 (Volumetric Floods)** (`DDOS-SYN_FLOOD`).
- **Hypothesized Failure Causes (Subject to Validation)**:
  - *Tier 4 Web Attacks*: We hypothesize that low precision and F1 in web attacks stem from the structural limitation of transport-layer and flow-statistic features (packet length counters, inter-arrival times, flag frequencies), which lack payload byte inspection necessary to disambiguate application-layer exploits (e.g., SQL syntax vs. benign HTTP requests).
  - *RECON-OSSCAN*: We hypothesize that operating system fingerprinting probes share statistical distributions with standard TCP connection anomalies and abnormal teardowns.
  - *DDOS-SYN_FLOOD*: We hypothesize that performance degradation is driven by feature-space collision with related SYN flood variants (`DDOS-SYNONYMOUSIP_FLOOD` and `DOS-SYN_FLOOD`).

---

## 2. Most Frequent Class-to-Class Misclassifications

Misclassification frequency was computed from the confusion matrix of balanced `fine_grained_lightgbm` on the internal test partition ($N = 75,105$).

### 2.1 Measured Confusion Frequencies

| True Ground Truth Class | Predicted Output Class | Misclassified Flows | % of True Class | Intra-Tier? |
| :--- | :--- | :---: | :---: | :---: |
| `DDOS-SYN_FLOOD` | `DDOS-SYNONYMOUSIP_FLOOD` | **1,245** | **47.34%** | **YES** (Tier 1) |
| `DDOS-TCP_FLOOD` | `DOS-TCP_FLOOD` | **1,140** | **46.51%** | **YES** (Tier 1) |
| `DOS-UDP_FLOOD` | `DDOS-UDP_FLOOD` | **1,112** | **40.84%** | **YES** (Tier 1) |
| `DOS-SYN_FLOOD` | `DDOS-SYNONYMOUSIP_FLOOD` | **934** | **33.90%** | **YES** (Tier 1) |
| `DOS-TCP_FLOOD` | `DDOS-TCP_FLOOD` | **670** | **25.50%** | **YES** (Tier 1) |
| `VULNERABILITYSCAN` | `SQLINJECTION` | **651** | **21.74%** | **NO** (Tier 3 $\rightarrow$ Tier 4) |
| `RECON-OSSCAN` | `BENIGN` | **625** | **21.06%** | **NO** (Tier 3 $\rightarrow$ Benign) |
| `RECON-OSSCAN` | `RECON-PORTSCAN` | **598** | **20.15%** | **YES** (Tier 3) |
| `DDOS-SYN_FLOOD` | `DOS-SYN_FLOOD` | **590** | **22.43%** | **YES** (Tier 1) |
| `DOS-SYN_FLOOD` | `DDOS-SYN_FLOOD` | **475** | **17.24%** | **YES** (Tier 1) |
| `DDOS-SYNONYMOUSIP_FLOOD` | `DDOS-SYN_FLOOD` | **464** | **18.87%** | **YES** (Tier 1) |
| `MITM-ARPSPOOFING` | `DNS_SPOOFING` | **432** | **15.64%** | **YES** (Tier 3) |
| `DNS_SPOOFING` | `MITM-ARPSPOOFING` | **414** | **13.63%** | **YES** (Tier 3) |
| `DDOS-UDP_FLOOD` | `DOS-UDP_FLOOD` | **411** | **15.60%** | **YES** (Tier 1) |
| `DDOS-SYNONYMOUSIP_FLOOD` | `DOS-SYN_FLOOD` | **403** | **16.39%** | **YES** (Tier 1) |

### 2.2 Measured Findings vs. Hypothesized Mechanisms
- **Measured Finding**: Ten of the top 12 off-diagonal confusion pairs (**$83.3\%$**) occur strictly between attack classes belonging to the **same functional threat tier**. Cross-tier misclassifications are concentrated in `VULNERABILITYSCAN` $\rightarrow$ `SQLINJECTION` (651 flows) and `RECON-OSSCAN` $\rightarrow$ `BENIGN` (625 flows).
- **Hypothesized Mechanisms (Subject to Validation)**:
  - *DDoS vs. DoS Confusion*: We hypothesize that DoS and DDoS variants of the same protocol (e.g., `DDOS-TCP_FLOOD` vs. `DOS-TCP_FLOOD`) share identical packet-level header fields, packet sizes, and flag patterns, differing primarily in source IP distribution and arrival rate dynamics that single-flow summary statistics capture imperfectly.
  - *Vulnerability Scan vs. SQL Injection*: We hypothesize that automated vulnerability scanners (e.g., Nessus, Nikto) actively probe HTTP endpoints by injecting SQL syntax strings, producing network flow statistics that closely mimic targeted SQL injection attacks.
  - *Spoofing Symmetry*: We hypothesize that ARP and DNS spoofing share similar local subnet broadcasting and anomalous resolution timing characteristics.

---

## 3. Attack Classes Frequently Misclassified as Benign (False Negatives)

A critical requirement of network intrusion detection is minimizing false negatives (attack traffic predicted as `BENIGN`). Across all 69,086 attack flows in the internal test partition, balanced `fine_grained_lightgbm` misclassified **1,517 flows ($2.20\%$)** as `BENIGN`.

### 3.1 Measured False Negatives to Benign

| Attack Class | Threat Tier | FN Count (to Benign) | Total Class Flows | % of Class as Benign |
| :--- | :---: | :---: | :---: | :---: |
| `RECON-OSSCAN` | Tier 3 (Recon) | **625** | 2,968 | **21.06%** |
| `RECON-PORTSCAN` | Tier 3 (Recon) | **253** | 2,950 | **8.58%** |
| `MITM-ARPSPOOFING` | Tier 3 (Recon) | **185** | 2,763 | **6.70%** |
| `VULNERABILITYSCAN` | Tier 3 (Recon) | **131** | 2,994 | **4.38%** |
| `RECON-HOSTDISCOVERY` | Tier 3 (Recon) | **113** | 2,931 | **3.86%** |
| `DNS_SPOOFING` | Tier 3 (Recon) | **86** | 3,037 | **2.83%** |
| `DICTIONARYBRUTEFORCE`| Tier 4 (Web) | **53** | 629 | **8.43%** |
| `BROWSERHIJACKING` | Tier 4 (Web) | **20** | 294 | **6.80%** |
| `RECON-PINGSWEEP` | Tier 3 (Recon) | **13** | 125 | **10.40%** |
| `COMMANDINJECTION` | Tier 4 (Web) | **11** | 255 | **4.31%** |
| `SQLINJECTION` | Tier 4 (Web) | **10** | 254 | **3.94%** |
| `BACKDOOR_MALWARE` | Tier 4 (Web) | **7** | 139 | **5.04%** |

### 3.2 Key Measured Proportions
- **Tier 3 Concentration**: Tier 3 Reconnaissance & Spoofing attacks account for **$80.4\%$ ($1,220$ of $1,517$ flows)** of all missed attacks.
- **Single Class Driver**: `RECON-OSSCAN` alone accounts for **$41.2\%$ ($625$ of $1,517$ flows)** of all false negatives, with **$21.06\%$** of its instances classified as benign.
- **Volumetric Flood Immunity**: Across Tier 1 (Volumetric Floods) and Tier 2 (Mirai Botnets), false negatives to benign traffic were $< 0.1\%$, indicating that flood-level traffic signatures are distinct from benign flow baselines.

---

## 4. Balanced vs. Unweighted LightGBM Error Patterns

Comparison between `fine_grained_lightgbm` (`class_weight='balanced'`) and `ablation_unweighted_fine_grained_lightgbm` (`class_weight=None`) on the internal test partition ($N = 75,105$).

### 4.1 Threat-Tier Performance Trade-offs

| Threat Tier / Metric | Balanced LightGBM | Unweighted LightGBM | Delta ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Overall Test Accuracy** | 72.65% | **74.45%** | **+1.80%** |
| **Overall Test Macro-F1** | 0.6145 | **0.6220** | **+0.0075** |
| **Total False Negatives (Attacks $\rightarrow$ Benign)** | **1,517 (2.20%)** | **3,418 (4.95%)** | **+1,901 (+125.3%)** |
| **Benign Recall** | 65.71% | **83.70%** | **+17.99%** |
| **Tier 1 (Volumetric Floods) Macro-F1** | **0.7943** | 0.7901 | -0.0042 |
| **Tier 1 Macro Recall** | **80.04%** | 79.20% | -0.84% |
| **Tier 2 (IoT Botnets) Macro-F1** | **0.9953** | 0.9752 | -0.0201 |
| **Tier 2 Macro Recall** | **99.41%** | 98.05% | -1.36% |
| **Tier 3 (Recon & Spoofing) Macro-F1** | 0.4696 | **0.4777** | +0.0081 |
| **Tier 3 Macro Recall** | 45.38% | **47.00%** | +1.62% |
| **Tier 4 (Web & Exploits) Macro-F1** | 0.1748 | **0.2203** | +0.0455 |
| **Tier 4 Macro Recall** | **34.63%** | **16.81%** | **-17.82%** |

### 4.2 Measured Tier 4 Recall Collapse Under Unweighted Optimization

| Attack Class | Balanced Recall | Unweighted Recall | Recall Delta |
| :--- | :---: | :---: | :---: |
| `SQLINJECTION` | **35.83%** | 7.48% | **-28.35%** |
| `BROWSERHIJACKING` | **53.06%** | 31.29% | **-21.77%** |
| `DICTIONARYBRUTEFORCE` | **52.62%** | 34.18% | **-18.44%** |
| `UPLOADING_ATTACK` | **23.33%** | 6.67% | **-16.67%** |
| `XSS` | **20.22%** | 3.83% | **-16.39%** |
| `BACKDOOR_MALWARE` | **25.18%** | 12.23% | **-12.95%** |
| `COMMANDINJECTION` | **32.16%** | 21.96% | **-10.20%** |

### 4.3 Interpretation
- **Measured Finding**: Unweighted training yields a higher aggregate accuracy (**$74.45\%$ vs. $72.65\%$**) by shifting the decision boundary in favor of dominant classes (Benign recall rises from $65.71\%$ to $83.70\%$).
- **Measured Consequence**: This accuracy improvement is accompanied by a **$+125.3\%$ surge in false negatives to benign traffic** ($1,517 \rightarrow 3,418$ missed flows) and a **$-17.82\%$ drop in Tier 4 Web Attack macro recall** ($34.63\% \rightarrow 16.81\%$). In minority classes such as `SQLINJECTION`, recall collapses from $35.83\%$ to $7.48\%$.

---

## 5. Strategy 1 vs. Strategy 2 LightGBM Error Patterns

Comparison between Strategy 1 (`fine_grained_lightgbm`, trained with collision vectors, $N_{\text{train}} = 350,821$) and Strategy 2 (`ablation_s2_fine_grained_lightgbm`, trained on collision-purged partitions, $N_{\text{train}} = 337,838$).

### 5.1 Measured Performance Comparison Across Partitions

| Metric / Evaluation Partition | Strategy 1 (Primary) | Strategy 2 (Purged) | Delta ($S2 - S1$) | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Internal Test Accuracy** | 72.65% | **72.69%** | **+0.04%** | Negligible difference |
| **Internal Test Macro-F1** | **0.6145** | 0.6140 | **-0.0005** | Comparable performance |
| **Internal Test MCC** | 0.7170 | **0.7175** | **+0.0005** | Negligible difference |
| **Internal Test False Negatives to Benign** | **1,517 (2.20%)** | 1,553 (2.25%) | **+36 (+0.05%)** | Comparable false negative rate |
| **Cross-Capture Accuracy** | 69.64% | **69.79%** | **+0.15%** | Marginal numerical gain |
| **Cross-Capture Macro-F1** | 0.6498 | **0.6511** | **+0.0013** | Marginal numerical gain |
| **Cross-Capture MCC** | 0.6861 | **0.6876** | **+0.0015** | Marginal numerical gain |

### 5.2 Threat-Tier Macro-F1 Comparison Across Partitions

| Threat Tier | S1 Internal F1 | S2 Internal F1 | Internal $\Delta$ | S1 Cross F1 | S2 Cross F1 | Cross $\Delta$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tier 1: Volumetric Floods** | **0.7943** | 0.7937 | -0.0006 | 0.8084 | **0.8099** | **+0.0015** |
| **Tier 2: IoT Botnets** | **0.9953** | 0.9952 | -0.0001 | 0.9930 | **0.9938** | **+0.0008** |
| **Tier 3: Recon & Spoofing** | 0.4696 | **0.4698** | +0.0002 | 0.4620 | **0.4636** | **+0.0016** |
| **Tier 4: Web & Exploit Attacks** | **0.1748** | 0.1731 | -0.0017 | 0.3253 | **0.3262** | **+0.0010** |

### 5.3 Interpretation of Strategy 2 Impact
- **In-Distribution Impact**: Purging collision vectors produces no meaningful change in in-distribution performance ($|\Delta| \le 0.0005$ across Macro-F1 and MCC).
- **Same-Testbed Cross-Capture Impact**: On the temporal cross-capture test set, Strategy 2 exhibits a small increase in Macro-F1 ($0.6498 \rightarrow 0.6511$, a difference of **$+0.0013$**), alongside minor gains across individual threat tiers ($+0.0008$ to $+0.0016$).
- **Conclusion**: These gains are **marginal in magnitude**. While removing contradictory label collisions did not harm performance and provided slight numerical improvements on temporal evaluation, the effect size is small and should not be characterized as proof of broader generalization capability.

---

## 6. Artifact & File References

- **Stage 4 Test Results Master Table**: [`results_v2/tables/stage4_test_results.csv`](tables/stage4_test_results.csv)
- **Detailed Model Stage 4 JSONs**:
  - Balanced Fine-Grained LightGBM: [`results_v2/metrics/stage4/fine_grained_lightgbm_stage4.json`](metrics/stage4/fine_grained_lightgbm_stage4.json)
  - Unweighted Fine-Grained LightGBM: [`results_v2/metrics/stage4/ablation_unweighted_fine_grained_lightgbm_stage4.json`](metrics/stage4/ablation_unweighted_fine_grained_lightgbm_stage4.json)
  - Strategy 2 Fine-Grained LightGBM: [`results_v2/metrics/stage4/ablation_s2_fine_grained_lightgbm_stage4.json`](metrics/stage4/ablation_s2_fine_grained_lightgbm_stage4.json)
- **Execution Script**: `scratch/stage5_error_analysis.py` (generates the full $34 \times 34$ confusion matrices and per-tier breakdowns from saved model checkpoints).
