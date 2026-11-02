# Capstone Review 3 Presentation Content

**Project Title**: IoT Network Attack Detection and Classification  
**Team**: Archon  
- Priyajit Biswal (23011102068)  
- Rohit K Manoj (23011102073)  
- Rishab Rajeev (23011102072)  
**Guide**: Dr. Vegesna S M Srinivasavarma, Associate Professor, Dept. of CSE  
**College**: Shiv Nadar University Chennai  

**Required Deliverables**:
1. Datasets Justification / Simulation details (10 Marks)
2. System architecture / Model (10 Marks)

**Figures Directory**: `snuc/review3_figures/`

---

## Slide 1: Title Slide

- **Layout**: Center Title + Two Cards at Bottom
- **Main Title**: IoT Network Attack Detection and Classification
- **Subtitle**: Capstone Project Phase 1 — Review 3
- **Card 1 (Team Archon)**:
  - Priyajit Biswal (23011102068)
  - Rohit K Manoj (23011102073)
  - Rishab Rajeev (23011102072)
- **Card 2 (Project Guide)**:
  - Dr. Vegesna S M Srinivasavarma
  - Associate Professor, Department of Computer Science & Engineering
  - Shiv Nadar University Chennai
- **Speaker Notes**:
  > "Respected Mentor and panel members, good afternoon. We are Team Archon. Today, we present our Phase 1 Review 3 progress on IoT Network Attack Detection and Classification, focusing on our dataset justification, preprocessing pipeline, and proposed multi-tier system architecture."

---

## Slide 2: Dataset Selection & Justification

- **Layout**: Two Columns (Left: Old Datasets vs. Right: CICIoT2023)
- **Title**: Dataset Selection & Justification
- **Subtitle**: Real-World IoT Testbed vs. Legacy Datasets
- **Column 1 — Limitations of Legacy Datasets**:
  - **NSL-KDD / KDD99**: Simulated Unix workstations from 1999; obsolete protocols, zero IoT relevance.
  - **BoT-IoT (2019)**: Over 99.9% synthetic floods; lacks realistic normal benign baseline.
  - **UNSW-NB15**: Standard enterprise office network; does not reflect constrained IoT edge gateways.
- **Column 2 — Why CICIoT2023 Benchmark?**:
  - **105 Physical Hardware Devices**: Real IoT hardware (smart TVs, webcams, smart bulbs), not simulated virtual machines.
  - **Protocol Heterogeneity**: 67 IP-active devices (Wi-Fi 802.11 b/g/n, Ethernet) + 38 Zigbee and Z-Wave smart devices bridged via 5 smart hubs.
  - **Non-Intrusive Hardware Capture**: Traffic mirrored using physical **Gigamon G-TAP A-Tx** taps and **Cisco Catalyst switches** without altering packet timing.
- **Picture to Insert**: *None*
- **Speaker Notes**:
  > "Legacy datasets like BoT-IoT are 99% synthetic floods with virtually no realistic normal baseline. We selected CICIoT2023 because it captures 105 physical devices across Wi-Fi, Ethernet, Zigbee, and Z-Wave through non-intrusive Gigamon hardware taps, yielding 548 GB of realistic network traffic."

---

## Slide 3: Traffic Capture & Features

- **Layout**: Text on Left + Picture on Right
- **Title**: Traffic Capture & Features
- **Subtitle**: Attack Taxonomy and Sliding-Window Flow Extraction
- **Left Column — Content**:
  - **Attack Diversity (33 Attacks in 7 Functional Groups)**:
    - **DDoS (12)**: SYN Flood, UDP Flood, SlowLoris, ICMP Flood, etc.
    - **DoS (4)**: TCP, UDP, SYN, HTTP Floods.
    - **Mirai Botnet (3)**: Mirai-GREETH, Mirai-GREIP, Mirai-UDPPlain.
    - **Reconnaissance (5)**: Port scans, OS scans, Host discovery, Ping sweep.
    - **Spoofing (2)**: ARP Spoofing and DNS Poisoning.
    - **Web Attacks (6)**: SQL Injection, Command Injection, XSS, Browser Hijacking.
    - **Brute Force (1)**: Dictionary password attacks against SSH/Telnet.
  - **Dual Sliding-Window Logic**:
    - **10-Packet Window**: Catches fast, short burst attacks (scans, spoofing, injection probes).
    - **100-Packet Window**: Tracks sustained, high-volume floods (DDoS/DoS campaigns).
  - **Verified 39-Feature Schema**:
    - Audited and verified all 63 CSV partitions to 39 numerical flow features: Packet sizes (`Min`, `Max`, `Avg`), timing dynamics (`Rate`, `IAT`), and Layer-4 flags (`TCP`, `UDP`, `SYN`, `ACK`).
- **Picture to Insert (Right Side)**:
  - File: `snuc/review3_figures/01_distribution_binary_and_8class.png`
  - Caption: Target class distributions across binary and 8-category triage.
- **Speaker Notes**:
  > "CICIoT2023 contains 33 attack vectors. Flow statistics are aggregated over dual sliding windows: 10 packets to catch stealthy burst exploits, and 100 packets for macro-level DDoS floods. We verified the ground-truth 39-feature schema across all 63 dataset partitions."

---

## Slide 4: Data Preprocessing & Sampling

- **Layout**: Text on Left + Picture on Right
- **Title**: Data Preprocessing & Sampling
- **Subtitle**: Preserving 100% of Rare Exploits While Handling Skew
- **Left Column — Content**:
  - **Big Data Challenge**: 46.6 million raw rows cause out-of-memory crashes. Handled using chunked streaming in 100,000-row blocks with `float32` precision.
  - **Extreme Natural Imbalance**: Volumetric floods make up $>85\%$ of raw traffic, while stealthy web attacks (SQLi, Command Injection) represent $<0.05\%$.
  - **Stratified Subsampling Strategy**:
    - Capped majority floods at 20,000 samples.
    - **Strictly retained 100% of all rare attacks**:
      - Command Injection: 1,666 (100% kept)
      - SQL Injection: 1,718 (100% kept)
      - Browser Hijacking: 1,904 (100% kept)
      - XSS: 1,239 (100% kept)
      - Backdoor Malware: 1,061 (100% kept)
      - Dictionary Brute Force: 4,181 (100% kept)
    - Final balanced working sample: **529,562 rows × 40 columns**.
  - **Zero-Leakage Cleaning**: Capped infinite packet rates, zero-imputed single-packet NaNs, and fitted `StandardScaler` strictly on the 80% training set.
- **Picture to Insert (Right Side)**:
  - File: `snuc/review3_figures/01_distribution_34class_stratified.png`
  - Caption: 34-class distribution in the 529k dataset showing 100% preservation of rare attacks.
- **Speaker Notes**:
  > "In natural traffic, floods make up 85% of flows, causing models to ignore rare web attacks. Instead of creating fake synthetic flows with SMOTE, our sampling capped floods while preserving 100% of every rare attack, giving us a clean, leak-free dataset of 529,000 samples across all 34 classes."

---

## Slide 5: Proposed System Architecture

- **Layout**: Architecture Diagram Centered at Top + 3 Cards at Bottom
- **Title**: Proposed System Architecture
- **Subtitle**: Three-Tier Hierarchical IoT Network Defense Model
- **Picture to Insert (Top Half)**:
  - File: `snuc/review3_figures/iot_defense_architecture.png`
  - Caption: End-to-end multi-tier defense architecture across border, edge, and cloud.
- **Bottom Content — The Three Tiers**:
  - **Tier 1: Wire-Speed Perimeter Filter (Border Switch)**:
    - *Model*: Online SGD Linear Classifier (log-loss).
    - *Performance*: **0.20 μs latency | >5,000,000 flows/sec**.
    - *Function*: Passes verified benign traffic at 10 Gbps line rate; forwards anomalies to Tier 2.
  - **Tier 2: Edge Gateway Triage (Embedded Router / Raspberry Pi)**:
    - *Model*: Reduced 20-Feature XGBoost Ensemble.
    - *Performance*: **12.15 μs latency | 82,299 flows/sec** (48.7% RAM savings).
    - *Function*: Drops volumetric floods immediately at the gateway; isolates botnets; escalates web exploits.
  - **Tier 3: Cloud Forensic Attribution (Centralized SIEM / SOC)**:
    - *Model*: Full 39-Feature 34-Class Classifier.
    - *Function*: Performs deep root-cause forensic classification across all 34 classes, mapping attacks to specific CVEs.
- **Speaker Notes**:
  > "Instead of a single slow deep learning model that drops packets at line rates, our architecture splits defense into three specialized tiers: Tier 1 screens normal traffic at wire speed in 200 nanoseconds; Tier 2 runs on the edge router dropping floods in 12 microseconds; only stealthy exploits reach Tier 3 in the cloud."

---

## Slide 6: System Workflow

- **Layout**: 5 Process Steps Horizontal Flow
- **Title**: System Workflow
- **Subtitle**: Step-by-Step Traffic Inspection and Remediation
- **Step 1: Packet Acquisition**:
  - Mirrored via non-intrusive Gigamon hardware taps and SPAN ports into dual sliding windows (10/100 packets).
- **Step 2: Real-Time Vectorization**:
  - 39 continuous flow features extracted and scaled in memory using baseline priors.
- **Step 3: Tier 1 Wire-Speed Filter (0.20 μs)**:
  - If Benign (Score > 0.95) $\to$ Passed immediately to local IoT devices with zero queue delay.
  - If Suspicious $\to$ Forwarded to Tier 2 edge queue.
- **Step 4: Tier 2 Edge Triage (12.15 μs)**:
  - If DDoS/DoS Flood $\to$ Dropped immediately at gateway perimeter.
  - If Botnet / Spoofing (Mirai) $\to$ Offending device MAC address quarantined.
  - If Complex Exploit $\to$ Escalated to Tier 3.
- **Step 5: Tier 3 Cloud Forensics**:
  - 34-class classification identifies exact attack vector and automatically updates upstream firewall / WAF rules.
- **Picture to Insert**: *None (Use clean flowchart or icons for the 5 steps)*
- **Speaker Notes**:
  > "This workflow ensures zero packet drops on legitimate traffic: normal IoT packets clear the switch in 200 nanoseconds, while edge and cloud processing are reserved strictly for anomalous traffic."

---

## Slide 7: Feature Selection & Optimization

- **Layout**: Text at Top + Two Pictures Side-by-Side at Bottom
- **Title**: Feature Selection & Optimization
- **Subtitle**: Consensus Feature Pruning for Resource-Constrained Edge Hardware
- **Top Text Content**:
  - **Tri-Model Consensus Scoring**:
    - Combines Gini impurity (Random Forest) + Information Gain (XGBoost) + Split Frequency (LightGBM).
    - **Top 5 Core Features (>90% Information)**: `Protocol Type` (11.2%), `syn_count` (5.8%), `Max packet size` (5.4%), `Header_Length` (5.3%), `Tot sum` (5.1%).
    - **Pruned Redundant Features (<0.1%)**: `IRC`, `DHCP`, `SMTP`, `Telnet`, `DNS`, `cwr_flag`, `ece_flag`, `LLC`, `IPv`, `IGMP`.
  - **Hardware Impact (39 Features vs. Reduced 20 Features)**:
    - **48.7% Memory Reduction**: Feature vector size drops from 156 bytes down to 80 bytes/flow.
    - **+28.55% Latency Speedup**: XGBoost inference drops from 17.01 μs to **12.15 μs (82,299 flows/sec)**.
    - **Minimal Accuracy Trade-off**: Accuracy delta is only -1.96% (74.12% $\to$ 72.16%).
- **Pictures to Insert (Bottom Half, Side-by-Side)**:
  - Left: `snuc/review3_figures/feature_importance_ranking.png` (Consensus rankings curve)
  - Right: `snuc/review3_figures/feature_reduction_comparison.png` (39 vs 20 features speedup comparison)
- **Speaker Notes**:
  > "To enable deployment on edge hardware like a Raspberry Pi, we combined feature importance across three tree models. Pruning 19 redundant protocol counters cut RAM usage by 48.7% and accelerated XGBoost by 28.5%, achieving 82,000 flows per second with less than 2% accuracy delta."

---

## Slide 8: Model Results & Benchmarks

- **Layout**: Table/Text on Left + Two Pictures Stacked/Side-by-Side on Right
- **Title**: Model Results & Benchmarks
- **Subtitle**: Validation Across Granularities and Minority Exploit Defense
- **Left Column — Benchmark Summary Table**:
  | Diagnostic Tier | Classes | Best Model | Accuracy | Latency | Throughput |
  | :--- | :---: | :--- | :---: | :---: | :---: |
  | **Tier 1 (Perimeter)** | 2 | **XGBoost** | **95.28%** | 1.05 μs | 954,681 fps |
  | **Tier 1 (Wire-Speed)**| 2 | **SGD LogReg** | **82.91%** | **0.20 μs** | **5,017,561 fps** |
  | **Tier 2 (Triage)** | 8 | **XGBoost** | **82.87%** | 6.24 μs | 160,318 fps |
  | **Tier 2 (Edge 20-Feat)**| 34 | **XGBoost** | **72.16%** | **12.15 μs** | **82,299 fps** |
  | **Tier 3 (Forensics)** | 34 | **XGBoost** | **74.13%** | 13.35 μs | 74,916 fps |
- **Minority Exploit Resolution (Zero Starvation via Cost-Sensitive Weights)**:
  - Command Injection: **81.11% Precision**
  - Dictionary Brute Force: **77.19% Precision**
  - Browser Hijacking: **71.19% Precision**
  - SQL Injection: **64.00% Precision**
- **Review 3 Deliverables Checklist**:
  - [x] Dataset Justification on 105 Physical Devices (Completed in Report Ch 3)
  - [x] Multi-Tier System Architecture & Model (Completed in Report Ch 4)
- **Pictures to Insert (Right Side)**:
  - Top: `snuc/review3_figures/04_attack34_minority_attacks_benchmark.png` (Rare attack precision bar chart)
  - Bottom: `snuc/review3_figures/category8_cm_xgboost.png` (8×8 confusion matrix)
- **Speaker Notes**:
  > "Our benchmarks validate our system architecture: Tier 1 runs at line-rate speed (5M flows/sec); Tier 2 runs on edge routers at 82,000 flows/sec; and our cost-sensitive loss weighting successfully eliminates minority exploit starvation, catching Command Injection with 81% precision and Brute Force with 77% precision."

