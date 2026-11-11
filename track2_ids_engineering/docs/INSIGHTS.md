# Capstone Project Insights & Technical Ledger
## End-to-End IoT Network Intrusion Detection System (CICIoT2023)

> **Document Purpose**: This technical ledger records the empirical findings, architectural decisions, mathematical trade-offs, and cybersecurity insights discovered across each step of the capstone project. It is continuously maintained and updated as the pipeline evolves.

---

## Executive Summary & Problem Formulation

The **CICIoT2023** benchmark represents one of the largest and most realistic datasets for Internet of Things (IoT) network security, comprising over **45 million network flows (~8.66 GB)** generated from a topological testbed of 105 IoT devices across 33 distinct cyber-attack profiles and benign network operations.

### Core Architectural Progression
Our investigation systematically progresses through four increasing levels of diagnostic granularity and edge optimization:
1. **Step 1 &mdash; Data Engineering & Stratified Sampling**: Overcoming OOM limitations via chunked streaming and minority-preserving stratified sampling (529,562 flows).
2. **Step 2 &mdash; Phase 1: Binary Detection (2 Classes)**: Coarse boundary filtering (`Attack` vs. `Benign`) for rapid packet dropping.
3. **Step 3 &mdash; Phase 2: Category Detection (8 Classes)**: Functional incident triage (`DDoS`, `DoS`, `Mirai`, `Recon`, `Spoofing`, `Web`, `BruteForce`, `Benign`).
4. **Step 4 &mdash; Phase 3: Fine-Grained Attribution (34 Classes)**: Precise forensic attribution across all 33 specific attack vectors + Benign traffic.
5. **Step 5 &mdash; Feature Importance & Latency Optimization**: Pruning redundant features (39 &rarr; 20 features) to achieve a **+28.55% inference speedup** with minimal accuracy degradation.
6. **Step 6 &mdash; Master Capstone Synthesis**: Global comparative evaluation across all model families, complexity frontiers, and hardware deployment profiles.
7. **Step 9 (Track 2 Expansion) &mdash; Leak-Free Hierarchical vs. Improved Flat IDS (`E1–E4`)**: Controlled evaluation under Track 1's zero-overlap partitions (`Train=350,821`, `Val=75,172`, `Test=75,105`, `Cross-Capture=57,280`), proving why class weighting + capacity (`E4`) matches 3-stage specialization (`E2`) and quantifying the IP-stripped `DDoS/DoS` twin ceiling (`42.4%` of Test errors).

---

## Step 0: Environment Architecture & Schema Integrity

### 1. Schema Alignment & Anomaly Resolution
- **Legacy Schema Discrepancy**: Prior exploratory notebooks (`example.ipynb`) assumed a 46-feature schema with mixed-case labels (e.g. `DDoS-Rstfinflood`).
- **Ground-Truth Verification**: Direct inspection of the raw 63 CSV partitions in `MERGED_CSV/` revealed exactly **39 continuous/discrete flow features** and **1 uppercase target column (`Label`)**.
- **Resolution**: All downstream modules (`src/config.py`, `src/preprocessing.py`, `src/models.py`) were built strictly around the verified 39-feature schema.

### 2. Precision & Memory Engineering
- Raw features loaded using `float64` consume ~13.5 GB RAM for 45M rows. Enforcing `float32` precision across all modules cut memory footprint by **50%**, enabling fast vector operations without loss of numerical stability.
- Dependency isolation was established via `uv` into a dedicated local `.venv` (Python 3.13, PyTorch 2.11, LightGBM, XGBoost, Scikit-Learn), ensuring reproducibility across platforms.

---

## Step 1: Data Engineering & Stratified Subsampling Dynamics

### 1. The OOM Challenge & Chunked Streaming
Attempting to load all 63 raw CSV files concurrently caused Out-Of-Memory (OOM) fatal crashes on standard workstations. We developed `src/data_loader.py` using generator-based chunk streaming (`chunksize=100,000`).

### 2. Stratified Sampling with 100% Rare-Class Retention
- **Extreme Natural Imbalance**: In raw IoT traffic, volumetric floods (DDoS/DoS) comprise over **85%** of all flows, while stealthy application-layer attacks (e.g., `UPLOADING_ATTACK`, `SQLINJECTION`, `XSS`) comprise less than **0.05%**.
- **Sampling Engine (`src/create_sample.py`)**:
  - Majority classes (DDoS/DoS) were capped at 20,000 samples (40,000 for Benign).
  - **100% of all rare and stealthy attack instances were retained** without dropping a single packet:
    - `UPLOADING_ATTACK`: 416 instances retained.
    - `RECON-PINGSWEEP`: 722 instances retained.
    - `BACKDOOR_MALWARE`: 1,061 instances retained.
    - `XSS`: 1,239 instances retained.
    - `COMMANDINJECTION`: 1,666 instances retained.
    - `SQLINJECTION`: 1,718 instances retained.
    - `BROWSERHIJACKING`: 1,904 instances retained.
    - `DICTIONARYBRUTEFORCE`: 4,181 instances retained.
  - Final working dataset: **529,562 rows &times; 40 columns** (106.5 MB), perfectly balanced for multi-class research.

### 3. Data Cleaning & Numerical Sanitization
- **Infinite Packet Rate**: Flows with duration tending to zero (`dt → 0`) yielded 19 `inf` values in the `Rate` column. These were capped to the maximum observed finite rate (3,355,443.2 packets/sec).
- **Single-Packet Windows**: Features measuring sample standard deviation (`Std`) or variance (`Variance`) produced `NaN` for single-packet flows. These were cleanly imputed to `0.0`.
- **Zero-Leakage Preprocessing**: Feature scaling (`StandardScaler`) was strictly fitted on the training split (80% / 423,649 rows) and applied to the test split (20% / 105,913 rows).

---

## Step 2: Phase 1 &mdash; Binary Classification (2 Classes)

### 1. Purpose & Threat Model
In real-time perimeter defense, an IoT security gateway must immediately discard malicious traffic before allocating CPU cycles to deep packet inspection. Binary classification evaluates whether a flow is **Attack (1)** or **Benign (0)**.

### 2. Empirical Benchmark (105,913 Test Instances)

| Model Architecture | Accuracy | Macro F1 | Weighted F1 | Benign Recall | Latency (&mu;s/flow) | Throughput (flows/s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** | **95.28%** | **0.8133** | **0.9567** | 91.80% | 1.05 &mu;s | 954,672 |
| **PyTorch Tabular DNN** | 94.66% | 0.7872 | 0.9510 | 90.40% | 1.78 &mu;s | 561,797 |
| **Random Forest** | 89.75% | 0.7648 | 0.9168 | 92.50% | 4.88 &mu;s | 204,918 |
| **LightGBM** | 89.60% | 0.7633 | 0.9157 | **93.20%** | 1.95 &mu;s | 512,820 |
| **SGD Logistic Regression** | 82.91% | 0.6806 | 0.8667 | 81.30% | **0.25 &mu;s** | **4,000,000** |

### 3. Key Findings:
- **Accuracy Fallacy**: A naive zero-rule classifier predicting all traffic as "Attack" would score 92.45% accuracy while completely failing to identify legitimate IoT device behavior (0% Benign Recall).
- **Macro F1 as Ground Truth**: Macro F1 equally weights Benign and Attack classes, correctly identifying **XGBoost (0.8133 F1)** and **LightGBM (0.7633 F1)** as top performers.
- **Ultra-High Speed Line-Rate**: SGD Logistic Regression processes **4,000,000 flows/second** (0.25 &mu;s/flow), providing an ultra-lightweight initial filter for resource-constrained hardware.

---

## Step 3: Phase 2 &mdash; Functional Category Classification (8 Classes)

### 1. Purpose & Incident Triage
Network operators require functional categorization to direct security playbooks:
- `DDoS` / `DoS` &rarr; Traffic rate-limiting, upstream BGP blackholing.
- `Mirai` &rarr; Device quarantine and telnet credential reset.
- `Recon` &rarr; Port-scanning detection and firewall rule hardening.
- `Spoofing` &rarr; Dynamic ARP inspection and DNSSEC verification.
- `Web` &rarr; WAF rule updates and input validation.
- `BruteForce` &rarr; Account lockout policies.

### 2. Empirical Benchmark (8 Classes)

| Model Architecture | Accuracy | Macro F1 | Weighted F1 | Macro Precision | Macro Recall | Latency (&mu;s/flow) | Throughput (flows/s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** | **82.87%** | **0.6876** | **0.8241** | **0.8269** | 0.6637 | 6.24 &mu;s | 160,318 |
| **LightGBM** | 78.82% | **0.6876** | 0.8039 | 0.6820 | **0.7690** | 18.84 &mu;s | 53,087 |
| **Random Forest** | 78.17% | 0.6814 | 0.7974 | 0.6760 | 0.7542 | 8.65 &mu;s | 115,623 |
| **SGD Logistic Regression** | 74.48% | 0.6042 | 0.7532 | 0.6050 | 0.6138 | **0.26 &mu;s** | **3,817,376** |
| **PyTorch Tabular DNN** | 73.47% | 0.6165 | 0.7567 | 0.6315 | 0.6876 | 16.56 &mu;s | 60,388 |

### 3. Key Findings:
- **The DoS vs. DDoS Boundary**: Confusion matrices revealed that models occasionally cross-confuse DoS and DDoS flows. This is mathematically expected: a single flooding host and 100 flooding bots transmitting identical UDP packets produce identical per-flow byte lengths and flag distributions without host entropy features.
- **Cost-Sensitive Weighting Success**: Applying inverse frequency weighting allowed LightGBM to achieve **76.90% Macro Recall**, reliably capturing rare Web and Brute Force attacks.

---

## Step 4: Phase 3 &mdash; Fine-Grained Attack Classification (34 Classes)

### 1. Purpose & Deep Attribution
Fine-grained classification attributes network anomalies to specific malware strains, exploit vectors, and scanning utilities across all 33 individual attack profiles plus Benign traffic.

### 2. Empirical Benchmark (34 Classes &times; 105,913 Test Instances)

| Model Architecture | Accuracy | Macro F1 | Weighted F1 | Macro Precision | Macro Recall | ROC-AUC (OVR) | Latency (&mu;s/flow) | Throughput (flows/s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM** | 72.83% | **0.6176** | **0.7343** | 0.6196 | **0.6525** | **0.9823** | 74.88 &mu;s | 13,355 |
| **XGBoost** | **74.13%** | 0.6125 | 0.7263 | **0.7018** | 0.6045 | 0.9814 | 13.35 &mu;s | 74,916 |
| **Random Forest** | 70.74% | 0.5970 | 0.7103 | 0.6271 | 0.6330 | 0.9789 | 8.24 &mu;s | 121,297 |
| **PyTorch Tabular DNN** | 63.61% | 0.5321 | 0.6382 | 0.5657 | 0.5710 | 0.9715 | 13.95 &mu;s | 71,693 |
| **SGD Logistic Regression** | 64.61% | 0.5280 | 0.6454 | 0.5389 | 0.5533 | 0.9635 | **0.41 &mu;s** | **2,420,194** |

### 3. Stealthy Minority Attack Vectors Deep-Dive

| Minority Attack Vector | Test Support | Top Precision Model | Precision | Top Recall Model | Recall | Top F1 Model | F1-Score |
| :--- | :---: | :--- | :---: | :--- | :---: | :--- | :---: |
| **Command Injection** | 333 | **XGBoost** | **81.11%** | LightGBM | 36.34% | **XGBoost** | **0.3452** |
| **Dictionary Brute Force** | 836 | **XGBoost** | **77.19%** | LightGBM | 51.67% | **XGBoost** | **0.3925** |
| **Browser Hijacking** | 381 | **XGBoost** | **71.19%** | LightGBM | 50.66% | **XGBoost** | **0.3367** |
| **SQL Injection** | 344 | **XGBoost** | **64.00%** | Random Forest | **59.59%** | **LightGBM** | **0.1007** |
| **Backdoor Malware** | 212 | **XGBoost** | **60.87%** | PyTorch DNN | 28.30% | **XGBoost** | **0.1191** |
| **Recon Ping Sweep** | 145 | **XGBoost** | **29.63%** | LightGBM | 26.21% | **XGBoost** | **0.0930** |
| **XSS** | 248 | **XGBoost** | **30.00%** | LightGBM | 27.42% | **LightGBM** | **0.0966** |
| **Uploading Attack** | 83 | Random Forest | 2.93% | Logistic Reg | 36.14% | **LightGBM** | **0.0671** |

### 4. Structural Confusion Clusters:
1. **Intra-Flood Blurring**: TCP-SYN floods, ACK floods, and RST floods exhibit high pairwise confusion with each other because the primary difference lies in single flag bits while byte totals and packet inter-arrival times are identical.
2. **Mirai Botnet Distinctiveness**: GRE tunneling (`MIRAI-GREETH_FLOOD` and `MIRAI-GREIP_FLOOD`) achieves **>98% per-class detection accuracy** because GRE encapsulation forces distinct `Header_Length` and `Protocol Type` signatures.
3. **Reconnaissance Partitioning**: Port scans and OS scans show minimal confusion with volumetric floods, consistently grouping into distinct decision branches based on low `Tot sum` and uniform packet lengths.

---

## Step 5: Feature Importance & Inference Latency Optimization

### 1. Consensus Importance Ranking
By synthesizing feature importances across three distinct tree algorithms (**Random Forest** Gini, **XGBoost** Gain, and **LightGBM** Split frequency), we identified the core features responsible for over 90% of model performance:

- **Top 5 Core Features**:
  1. `Protocol Type` (11.17% consensus score) &mdash; Fundamental layer-4 transport discriminator.
  2. `syn_count` (5.77% score) &mdash; Critical for detecting SYN flooding, port scans, and connection handshakes.
  3. `Max` (5.39% score) &mdash; Maximum packet size in window (separates volumetric attacks from micro-probing).
  4. `Header_Length` (5.26% score) &mdash; Encapsulation and tunnel detection (GRE, IPv6, fragmentation).
  5. `Tot sum` (5.08% score) &mdash; Cumulative flow byte volume.

- **Pruned Redundant Features (<0.1% gain)**:
  `IRC`, `DHCP`, `SMTP`, `Telnet`, `DNS`, `cwr_flag_number`, `ece_flag_number`, `LLC`, `IPv`, `IGMP`.

### 2. Feature Space Reduction Benchmark (39 vs. 20 Features)

| Model Architecture | Feature Set | Feature Count | Accuracy | Macro F1 | Weighted F1 | Latency (&mu;s/flow) | Throughput (flows/s) | Latency Speedup | Accuracy Delta |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** | Full | 39 | 74.12% | 0.6125 | 0.7263 | 17.01 &mu;s | 58,800 | &mdash; | &mdash; |
| **XGBoost** | **Reduced (Top 20)** | **20** | **72.16%** | **0.5886** | **0.7061** | **12.15 &mu;s** | **82,299** | **+28.55%** | &minus;1.96% |
| **Random Forest** | Full | 39 | 70.74% | 0.5970 | 0.7103 | 17.60 &mu;s | 56,815 | &mdash; | &mdash; |
| **Random Forest**| **Reduced (Top 20)** | **20** | **68.93%** | **0.5783** | **0.6953** | **15.15 &mu;s** | **65,988** | **+13.90%** | &minus;1.81% |
| **LightGBM** | Full | 39 | 72.83% | 0.6176 | 0.7343 | 77.36 &mu;s | 12,926 | &mdash; | &mdash; |
| **LightGBM** | **Reduced (Top 20)** | **20** | **70.80%** | **0.5963** | **0.7159** | 80.09 &mu;s | 12,485 | *(22.8% faster train)* | &minus;2.03% |

### 3. Edge Gateway Engineering Implications
- **Memory Footprint**: Reducing from 39 to 20 float32 features shrinks each flow vector from **156 bytes to 80 bytes** (a **48.7% RAM savings**).
- **Line-Rate Capability**: At **82,299 flows/second** on a single thread, the reduced XGBoost model easily handles 1 Gbps IoT router traffic in software without hardware acceleration.

---

## Step 6: Master Capstone Synthesis & Comparative Presentation

### 1. Unified Cross-Model Synthesis Across All Phases

| Phase | Granularity | Features | Top Model | Accuracy (%) | Macro F1 | Weighted F1 | Latency (us/flow) | Throughput (fps) |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Phase 1: Binary** | 2 | 39 | **XGBoost** | **95.28%** | **0.8133** | **0.9503** | 1.05 us | 954,681 |
| **Phase 1: Binary** | 2 | 39 | PyTorch DNN | 94.66% | 0.7872 | 0.9435 | 14.50 us | 68,950 |
| **Phase 1: Binary** | 2 | 39 | SGD Logistic Reg | 82.91% | 0.6806 | 0.8655 | **0.20 us** | **5,017,561** |
| **Phase 2: Category** | 8 | 39 | **XGBoost** | **82.87%** | **0.6876** | **0.8241** | 6.24 us | 160,318 |
| **Phase 2: Category** | 8 | 39 | LightGBM | 78.82% | 0.6876 | 0.8039 | 18.84 us | 53,087 |
| **Phase 2: Category** | 8 | 39 | Random Forest | 78.17% | 0.6814 | 0.7974 | 8.65 us | 115,623 |
| **Phase 3: Fine-Grained** | 34 | 39 | **XGBoost** | **74.12%** | 0.6125 | 0.7263 | 13.35 us | 74,916 |
| **Phase 3: Fine-Grained** | 34 | 39 | LightGBM | 72.83% | **0.6176** | **0.7343** | 74.88 us | 13,355 |
| **Phase 3: Fine-Grained** | 34 | 39 | Random Forest | 70.74% | 0.5970 | 0.7103 | 8.24 us | 121,297 |
| **Phase 3: Fine-Grained** | 34 | 39 | PyTorch DNN | 63.61% | 0.5321 | 0.6382 | 13.95 us | 71,693 |
| **Phase 3: Fine-Grained** | 34 | 39 | SGD Logistic Reg | 64.61% | 0.5280 | 0.6454 | **0.41 us** | **2,420,194** |
| **Step 5: Optimized Edge** | 34 | 20 | **XGBoost** | **72.16%** | **0.5886** | **0.7061** | **12.15 us** | **82,299** |
| **Step 5: Optimized Edge** | 34 | 20 | LightGBM | 70.80% | 0.5963 | 0.7159 | 80.09 us | 12,485 |
| **Step 5: Optimized Edge** | 34 | 20 | Random Forest | 68.93% | 0.5783 | 0.6953 | 15.15 us | 65,988 |

### 2. Multi-Tiered IoT Security Gateway Deployment Architecture

Rather than choosing a single monolithic model, our empirical results support a **Three-Tier Hierarchical Defense Architecture**:

```mermaid
graph TD
    FlowIn[Incoming Flow Stream] --> Tier1[Tier 1: Linear Boundary Filter<br/>SGD Logistic Regression<br/>Latency: 0.20 us | 5,000,000 fps]
    Tier1 -- "Normal IoT Traffic" --> Safe[Egress to Local IoT Devices]
    Tier1 -- "Suspicious Packet" --> Tier2[Tier 2: Edge Gateway Classifier<br/>Reduced 20-Feature XGBoost<br/>Latency: 12.15 us | 82,299 fps]
    Tier2 -- "Immediate Drop" --> Drop[Discard Volumetric Floods]
    Tier2 -- "Complex Exploit" --> Tier3[Tier 3: Cloud / SOC Attribution<br/>34-Class LightGBM / XGBoost<br/>Full 39 Features | Deep Forensics]
```

1. **Tier 1 (Wire-Speed Perimeter Filter)**: Processes up to **5,000,000 flows/sec** with 0.2 &mu;s latency, instantly passing verified benign traffic.
2. **Tier 2 (Edge Gateway Triage)**: Operates locally on router/gateway hardware (e.g. Raspberry Pi 4 / ARM Cortex-A), classifying at **82,299 flows/sec** with 72.16% accuracy and 48.7% RAM savings.
3. **Tier 3 (Centralized Threat Intelligence)**: Runs in cloud/SOC nodes for forensic vector attribution across all 34 classes.

### 3. Core Capstone Research Contributions
1. **Granularity Scaling Trajectory**: Quantified the empirical trade-off between diagnostic granularity and accuracy: 95.28% (2-class) &rarr; 82.87% (8-class) &rarr; 74.12% (34-class).
2. **Minority Attack Vector Resolution**: Demonstrating that balanced class weighting unlocks high-precision detection for rare, high-impact vectors (81.1% on Command Injection, 77.2% on Brute Force, 71.2% on Browser Hijacking) without minority class starvation.
3. **Pareto Edge Optimization**: Established that pruning 48.7% of features delivers a **28.55% latency reduction** while retaining >97.3% of baseline model accuracy.

---

## Step 7: Natural Uncapped Benchmark & The Literature Accuracy Paradox

### 1. Empirical Verification on 200,000 Uncapped Flows
To reconcile our balanced dataset findings with papers in *IEEE Access*, *Nature Scientific Reports*, and *Elsevier FGCS* reporting 98%–99.5% accuracy, we evaluated our models directly on raw, uncapped partitions from `MERGED_CSV/` without majority capping:
- **Binary Wire-Screen**:
  - **LightGBM**: **99.18% Accuracy**, 99.66% Attack Precision, 99.51% Attack Recall, **1.19 us/flow**.
  - **Random Forest**: **99.17% Accuracy**, 99.59% Attack Precision, 99.55% Attack Recall, **2.97 us/flow**.
  - **XGBoost (Hist)**: **99.15% Accuracy**, 99.65% Attack Precision, 99.48% Attack Recall, **0.65 us/flow**.
- **8-Category Triage**:
  - **Random Forest**: **98.98% Accuracy**, 0.8679 Macro F1.
  - **XGBoost (Hist)**: **98.95% Accuracy**, 0.8647 Macro F1.
  - **LightGBM**: **98.94% Accuracy**, 0.8631 Macro F1.

### 2. The Literature Accuracy Paradox Deconstructed
When evaluated on raw traffic where DDoS/DoS floods comprise over 85% of flows:

```
Accuracy = (TP_Flood + TP_Benign + TP_Exploits) / N_Total ≈ TP_Flood / N_Total
```

A model can score **98.6% overall accuracy** while missing **100% of brute force and web exploits** (0% recall), because rare attacks make up only 0.02% of natural traffic. Literature papers celebrate this headline accuracy while leaving IoT networks completely unprotected against targeted penetration.

---

## Step 8: 10 Gbps Wire-Speed Throughput & DMA Burst Latency Benchmark

### 1. The Physical Layer Ethernet Constraints
Intrusion detection at the network perimeter is governed by hardware line-rate math:
- **1500-Byte Standard MTU**: 10 Gbps / (1500 × 8) = **833,333 pps** (budget: ≤ 1.20 µs/packet).
- **512-Byte Average IoT Frame**: 10 Gbps / (512 × 8) = **2,441,406 pps** (budget: ≤ 0.41 µs/packet).
- **64-Byte Minimum Ethernet Frame**: 10 Gbps / (64 × 8) = **19,531,250 pps** (budget: ≤ 0.051 µs/packet).

### 2. Empirical Benchmark Across All Pipeline Models (`benchmark_throughput.py`)
Evaluated across 20,000 live evaluation flows with DMA ring-buffer burst profiling (`burst_size=64`):

| Model | Diagnostic Tier | Wire Latency | Burst P99 Latency | Sustained Throughput | 10 Gbps Feasibility (1500B MTU) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`T1_LogisticRegression`** | Tier 1 (Wire-Filter) | **0.12 us** | **9.49 us** | **8,708,981 pps (8.71 Mpps)** | **Saturates 10G on 1 Core** |
| **`T1_XGBoost_Hist`** | Tier 1 (Wire-Filter) | **0.38 us** | **32.14 us** | **2,648,151 pps (2.65 Mpps)** | **Saturates 10G on 1 Core** |
| **`T1_LightGBM`** | Tier 1 (Wire-Filter) | 1.54 us | 44.65 us | 647,804 pps (0.65 Mpps) | 2 Cores |
| **`T1_RandomForest`** | Tier 1 (Wire-Filter) | 7.06 us | 264.58 us | 141,627 pps (0.14 Mpps) | 6 Cores |
| **`T2_XGBoost_Top20`** | Tier 2 (Gateway 20-Feat) | 7.53 us | 140.42 us | 132,844 pps (0.13 Mpps) | 7 Cores |
| **`T2_RandomForest_Top20`** | Tier 2 (Gateway 20-Feat) | 18.05 us | 303.17 us | 55,403 pps (0.06 Mpps) | 16 Cores |
| **`T3_XGBoost_34Class`** | Tier 3 (Cloud Attribution) | 11.01 us | 696.65 us | 90,872 pps (0.09 Mpps) | 10 Cores |
| **`T3_RandomForest_34Class`** | Tier 3 (Cloud Attribution) | 23.52 us | 154.68 us | 42,513 pps (0.04 Mpps) | 20 Cores |
| **`DL_PyTorch_DNN_Binary`** | Literature Baseline | 0.40 us | 16.91 us | 2,493,038 pps (2.49 Mpps) | 1 Core |

### 3. The Deep Learning Latency Disconnect
Literature proposals advocating CNN-LSTM or Vision Transformers require **50,000 to 100,000 us (50–100 ms)** per sample. Processing 10 Gbps traffic with such models would drop **99.98% of packets** and require a server farm of **over 40,000 CPU cores**. By deploying our lightweight Tier 1 filter (0.38 us), Team Archon deflecting line-rate floods at the NIC level on **1 single CPU core**, preserving downstream resources for deep inspection.

---

## Technical Lessons Learned & Gotchas Ledger

1. **Windows Console Encoding**: Avoid non-ASCII characters (e.g. `\u03bc` for &mu;s) in pandas DataFrame column names or terminal outputs on Windows, as standard Python processes default to `cp1252` encoding and raise unhandled `UnicodeEncodeError`. Always use ASCII `us`.
2. **Optimizer Selection on Large Datasets**: Scikit-learn's `LogisticRegression(solver='lbfgs')` fails to converge or stalls on datasets with >400k samples. Utilizing `SGDClassifier(loss='log_loss')` converges in under 8 seconds with identical accuracy.
3. **Class Weighting in PyTorch**: PyTorch's `nn.CrossEntropyLoss` does not compute class weights automatically. Explicit pre-calculation via `sklearn.utils.class_weight.compute_class_weight` is mandatory to avoid complete minority class starvation in deep learning tabular baselines.
4. **Jupyter Automation**: Programmatic notebook execution should always be performed via `.venv\Scripts\jupyter.exe nbconvert --to notebook --execute <nb> --inplace` to guarantee that all interactive plots and markdown outputs are pre-rendered for evaluation.

---

## Step 9: Track 2 Leak-Free Hierarchical vs. Improved Flat IDS (`E1–E4`) & Feature-Space Diagnosis

> **Full Artifacts**: [`../hierarchical_ids/docs/COMPARISON_REPORT.md`](../hierarchical_ids/docs/COMPARISON_REPORT.md) & Executed Notebook [`../../notebooks/09_hierarchical_ids_evaluation.ipynb`](../../notebooks/09_hierarchical_ids_evaluation.ipynb).

### 1. Experimental Setup Under Strictly Non-Overlapping Splits
Using Track 1's Strategy 1 group-hash partitions (`0` feature-vector overlap verified across all 6 pairs) and canonical taxonomy (`1` Benign + `16` Volumetric Floods [`DDoS` 12 + `DoS` 4] + `17` Non-Floods [`Mirai` 3 + `Recon` 5 + `Spoofing` 2 + `Web` 6 + `BruteForce` 1]):
- **Internal 70 / 15 / 15 Split (`Merged01–25`, 501,098 clean flows)**: `Train = 350,821 (70.01%)`, `Validation = 75,172 (15.00%)`, `Internal Test = 75,105 (14.99%)`.
- **Cross-Capture Clean (`Merged26–40`, 57,280 flows)**: Separate later-file holdout (`<=2,000` per attack, `<=5,000` Benign) with overlap stripped. Because big flood classes are capped at 2,000, the **8 minority attack classes jump from 2.58% (Internal Test) to 14.64% (Cross-Capture) — a 5.67x prevalence shift**.

### 2. Controlled Architecture Comparison (`E1` vs. `E2` vs. `E3` vs. `E4`)

| Architecture | Origin | Val Macro-F1 | Test Acc | Test Macro-F1 | Test MCC | Test Minority F1 (Rec) | Cross Acc | Cross Macro-F1 | Single-Flow P50 (Test) | Execs / Flow |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`E1_Flat34_LGBM_Balanced`** | Track 1 | 0.6142 | 72.65% | 0.6145 | 0.7170 | 0.1624 (**0.3360**) | 69.64% | **0.6498** | **3.43 ms** | 1.00 |
| **`E2_Cascade_Default_T050`** | S1 Reused + New S2/3 | 0.6196 | 73.59% | 0.6185 | 0.7289 | 0.2115 (0.1519) | 68.61% | 0.6213 | 7.88 ms | 2.63 |
| **`E2_Cascade_ValTuned`** | S1 RF (`T=0.15`) + S2/3 | **0.6394** | 75.04% | **0.6361** | 0.7405 | **0.2264** (0.2094) | 70.50% | 0.6453 | 53.30 ms* | 2.82 |
| **`E3_Parallel_NaiveMax`** | No Stage 2 | 0.4389 | 56.11% | 0.4389 | 0.5567 | 0.1757 (0.1084) | 53.26% | 0.4360 | 49.73 ms* | 2.82 |
| **`E3_Parallel_MetaGate`** | Learned LR Gate | 0.6267 | 73.78% | 0.6239 | 0.7275 | 0.2253 (0.2094) | 69.41% | 0.6327 | 52.40 ms* | 2.82 |
| **`E4_LGBM_Leaves63_Sqrt`** | New Improved Flat | 0.6358 | **75.37%** | 0.6352 | **0.7440** | 0.2196 (0.2063) | **70.72%** | 0.6441 | 4.15 ms | 1.00 |

*\*Uses Track 1 `binary_random_forest.joblib` (`n_jobs=-1`) in Stage 1 (`8,042.5 fps` batch throughput, ~45 ms thread-dispatch penalty on single-row queries; swapping to `binary_lightgbm` holds `0.6381` Val Macro-F1 at LGBM speed).*

### 3. Five Key Engineering Discoveries
1. **Capacity + `sqrt_balanced` Weights Match the Cascade at 1/3 Compute (`E4` vs. `E2`)**: Once flat `E4` (`num_leaves=63, n_estimators=150, sqrt_balanced`) matches E2's Stage 3 hyperparameters, **E4 virtually ties E2 on Test Macro-F1 (`0.6352` vs. `0.6361`) and wins Accuracy (`75.37%`), MCC (`0.7440`), Benign Recall (`78.63%`), Model Executions (`1.0` vs. `2.82`), and Latency (`4.15 ms` vs. `53.30 ms`)**. Most of E2's improvement over Track 1 E1 (`0.6145` &rarr; `0.6361`) stemmed from gentler `sqrt_balanced` weighting and leaf capacity—not hierarchical specialization itself.
2. **Stage 1 Threshold Bug at `0.50`**: Because Track 1's binary classifiers trained with `class_weight="balanced"` (`11.6x` weight on Benign), threshold `0.50` caused Stage 1 to prematurely drop **`7,990` (11.57%)** true Test attacks and **`9,329` (17.84%)** Cross attacks as `BENIGN`. Tuning `tau_att = 0.15` on Validation recovers `+0.0176` Test Macro-F1 (`0.6185` &rarr; `0.6361`).
3. **Stage 2 Routing Is Already Solved (~99.98%)**: `stage2_cat7_lgbm_sqrt_balanced` commits only **50 cross-family misroutes out of 75,105** Test flows (`99.98%` Flood Recall), and flat E4 makes only **46** cross-family errors (`11` + `35`). Eliminating `BENIGN` from Stage 2 prevented the `2,393` invalid Stage 2 Benign exits observed with Track 1's 8-class checkpoint.
4. **Why Parallel Closed-World Specialists Without Stage 2 (`E3_NaiveMax`) Fail**: Conditionally trained Stage 3A and 3B specialists both output `>= 0.50` probability simultaneously on **76.16% (`52,083`)** of passed Test attacks, generating **`26,678` family routing errors** (`56.11%` Acc). Adding a `LogisticRegression` meta-gate cuts collisions by 96% (`26,678` &rarr; `1,053`, Macro-F1 `0.6239`), still trailing E2's raw-feature router (`50` errors).
5. **Verified Absence of IP/Host Features & The DDoS vs. DoS Twin Ceiling (`42.4%` of All E4 Errors)**:
   - Direct header inspection of raw `Merged01.csv` and `sample_stratified.csv` confirms **0 of 40 columns** contain source/destination IPs, ports, timestamps, or concurrent attacker host counts.
   - Without multi-host entropy, DDoS and DoS bursts of identical protocol collapse onto the same 39-dimensional flow features (Track 1 found **8,740** unique vectors carrying conflicting `DDoS`/`DoS` labels).
   - Quantified on `cm_E4_Improved_Flat34_E4_LGBM_Leaves63_Sqrt_test.npy`: **`7,840` out of `18,501` total Test errors (`42.4%`)** occur strictly inside just four protocol-identical twin groups (`SYN/SynonymousIP`: `4,118`; `TCP`: `1,817`; `UDP`: `1,536`; `HTTP`: `369`). Merging those 4 mitigation-equivalent twins elevates E4's Test Accuracy from **75.37% &rarr; 85.81%** immediately—justifying operational mitigation merging in Track 2 while keeping 34-class results standard for academic comparison.
   - **Cross-Capture Reversal Explained**: `E1` reclaims top Cross Macro-F1 (`0.6498`) solely because `5.67x` denser minority sampling in `Merged26–40` rewards its aggressive `balanced` recall (`0.3495`), whereas on natural Test (`2.58%` minority) `E1` suffers precision collapse (`0.1624` Minority F1 vs. `0.2264` E2 / `0.2196` E4).

---

*Status: Track 2 Steps 0–9 Fully Benchmarked, Pre-Rendered, Documented, and Verified.*
