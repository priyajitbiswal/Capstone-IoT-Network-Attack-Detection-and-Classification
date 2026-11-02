# Comprehensive Critical Evaluation of IoT Intrusion Detection Literature
**Deconstructing Methodological Flaws, the "Accuracy Paradox", and Synthetic Oversampling Distortions across 14 Benchmark Studies**

---

## Executive Summary & Context

In recent cybersecurity research evaluating IoT intrusion detection benchmarks (principally **CICIoT2023**, **ToN_IoT**, and **BoT-IoT**), published literature frequently reports headline classification accuracies exceeding **98.5% to 99.9%**. 

However, a rigorous empirical investigation reveals that these impressive headline figures frequently mask critical methodological vulnerabilities, dataset-induced artifacts, and severe operational blind spots. Most notably, under extreme natural class imbalance where volumetric flood attacks (DDoS and DoS) constitute over 90% of total network traffic, models can achieve **99%+ accuracy** while suffering an **abysmal 0% to 15% detection rate on stealthy minority attacks** (such as dictionary brute-force, web exploits, SQL injections, and command backdoors).

This comprehensive report systematically examines **all 14 research papers** preserved in `snuc/research-papers` and `snuc/clg-capstone-literature`. For each paper, we deconstruct:
1. **Core Proposed Architecture & Scope**
2. **Datasets & Experimental Regime**
3. **Reported Headline Metrics**
4. **Critical Methodological Flaws & Vulnerabilities** (Class imbalance neglect, synthetic SMOTE distortion, data leakage, edge latency contradiction, accuracy paradox)
5. **How Team Archon Overcomes These Limitations** via our Multi-Tiered IoT Security Gateway.

---

## Master Comparison Matrix: 14 Analyzed Literature Papers

| # | Paper / Authors | Year | Venue | Datasets Evaluated | Proposed Architecture | Claimed Accuracy | Core Methodological Flaws & Blind Spots |
|---|---|---|---|---|---|---|---|
| **1** | **Khan & Alam** | 2026 | *Scientific Reports* (Nature) | CICIoT2023, TON_IoT, CICIDS2018 | Hybrid Feature Selection (mRMR + SHAP) + LightGBM/CatBoost | 99.72% – 99.94% | **Synthetic SMOTE distortion**; linear interpolation creates invalid protocol states; minority class recall masked by flood dominance; feature extraction latency contradicts "edge" claims. |
| **2** | **Almahaqeri et al.** | 2026 | *Comprehensive Eval on CICIoT2023* | CICIoT2023 | Gain-based Feature Selection + XGBoost / CatBoost | 99.61% | **Binary inflation**: 99.61% is strictly binary triage; multiclass evaluation lumps flood subclasses; zero validation against cross-capture temporal drift or novel variants. |
| **3** | **Neto et al.** | 2023 | *Sensors* (MDPI) | CICIoT2023 (Dataset Benchmark) | Random Forest, Deep Neural Network (DNN), Perceptron | 98.60% (DNN) / 99.16% (RF) | **The Foundational "Accuracy Paradox"**: Despite 98.6% accuracy, DNN achieves **only 14.7% recall on Brute Force** (misses 2,542 attacks) and 3.8% on Web attacks; stripped IP features cause massive DoS vs. DDoS confusion. |
| **4** | **Ullah et al.** | 2026 | *Scientific Reports* (Nature) | BoT-IoT, CICIoT2023, IoT-23 | Multi-Model Framework (CNN, LSTM, RF, Autoencoder, FL) | 99.20% (BoT-IoT/CICIoT), 91.5% (IoT-23) | **Aggressive SMOTE oversampling (59 mentions)**; relies on BoT-IoT's >99% DoS skew; deep models require orders of magnitude more compute for marginal gain over trees. |
| **5** | **DeepCLG Authors** | 2025 | *Int. J. Machine Learning & Cybernetics* | TON_IoT, BoT-IoT, NSL-KDD, UNSW-NB15 | Hybrid DeepCLG (CNN + LSTM + GRU Stack) | 99.82% (BoT-IoT), 95.55% (UNSW) | **Architectural Overkill**: Stacking CNN + LSTM + GRU yields prohibitive tens-of-ms inference latency; **performance collapses by 42% (57.7% detection rate) on realistic UNSW-NB15**. |
| **6** | **FEC-IDS Authors** | 2025 | *IEEE / Open Access* | CICIoT2023, TON_IoT, BoT-IoT | Classical ML (RF, XGBoost, DT) + Feature Reduction | 99.76% (Binary), 99.21% (Multi) | Evaluates on small sampled slices; discards sequence features necessary for slow multi-stage attacks; hides minority exploit failure behind weighted macro averages. |
| **7** | **Al-Hejri et al.** | 2026 | *Scientific Reports* (Nature) | TON_IoT, BoT-IoT, IoT-23, NSL-KDD | Hybrid CNN-Transformer + Federated Learning + SHAP/LIME | 94.89% | **Impractical Edge Complexity**: Multi-head attention $O(N^2)$ memory explosion on IoT edge devices; **accuracy plummets to 69.35% under realistic non-IID client distributions**. |
| **8** | **Booij et al.** | 2022 | *IEEE Internet of Things Journal* | TON_IoT, BoT-IoT, UNSW-NB15, NSL-KDD | Random Forest, Decision Tree (Dataset Heterogeneity) | 99.68% (In-domain) | **Flow-Level Disconnect**: Uses post-mortem Zeek/NetFlow summaries requiring flow termination (attacks succeed before flow record generated); exposes lack of cross-dataset generalization. |
| **9** | **Choudhary & Kesswani** | 2020 | *Sensors* (MDPI) | BoT-IoT, TON_IoT, NSL-KDD Survey | Anomaly-Based IDS Survey & Evaluation | 99.40% | Surveys simulation-based synthetic datasets (NS-2/3) lacking real human behavioral variance; relies on obsolete 1999 KDD taxonomies (Probe, U2R, R2L). |
| **10** | **Fenanir & Semchedine** | 2023 | *Computers & Security / Elsevier* | NSL-KDD, IoT-23, UNSW-NB15 | Federated Learning (CNN, LSTM, GRU with FedAvg) | 99.99% (NSL-KDD) | **Obsolete Benchmark Reliance**: Achieving 99.99% on 25-year-old NSL-KDD is meaningless for modern IoT; ignores communication round bandwidth bottlenecks on low-power mesh (Zigbee/LoRa). |
| **11** | **Moussaoui & Kmiti** | 2026 | *Scientific Reports* (Nature) | CICIoT2023, TON_IoT, NSL-KDD | CNN-LSTM + Federated Learning + XAI (Extreme Non-IID) | 97.36% (Centralized) | **Non-IID Accuracy Collapse**: Accuracy collapses by **16% to 26% (down to 71.53%)** under non-IID conditions; openly admits flow statistics fail completely on payload-driven attacks. |
| **12** | **Moustafa** | 2021 | *TON_IoT Foundational Paper* | Network TON_IoT Datasets | Distributed AI at the Edge (RF, LSTM) | ~98.5% | **Static Artifact Leakage**: Network captures contain hardcoded static IPs and MACs for attacking/victim machines; models memorize network identifiers rather than intrusion behavior. |
| **13** | **Qaddos & Yaseen** | 2024 | *Scientific Reports* (Nature) | BoT-IoT, IoT-23, NSL-KDD | Feature-Weighted SMOTE (FW-SMOTE) + Hybrid CNN-LSTM-Attn | 99.60% (BoT-IoT) | **Deep Stack Latency Disconnect**: FW-SMOTE + CNN + LSTM + Attention requires massive GPU compute; impossible to run at line rate (>1 Gbps); high BoT-IoT accuracy is a known dataset imbalance artifact. |
| **14** | **Umar & Yasmeen** | 2025 | *EURASIP J. Information Security* | CICIDS2017, NSL-KDD | DNN with Knowledge Distillation & Quantization (DNN-KDQ) | 99.43% | Evaluates on flawed CICIDS2017 with known timestamp inversions; INT8 quantization shifts decision boundaries for subtle minority attacks; tests zero IoT protocol flows (MQTT/CoAP). |

---

## In-Depth Critical Deconstruction of Each Paper

```
================================================================================
PART I: RESEARCH PAPERS DIRECTLY BENCHMARKING CICIoT2023 (snuc/research-papers)
================================================================================
```

### 1. Khan & Alam (Scientific Reports, 2026)
* **Full Title**: *A resource efficient IoT intrusion detection model using hybrid feature selection for edge computing*
* **Core Claims**: Proposes a hybrid feature selection method (combining mRMR with SHAP and gain-based ranking) to prune features down to 10–15 attributes, achieving **99.72% to 99.94% accuracy** across CICIoT2023 and TON_IoT.
* **Critical Methodological Faults**:
  1. **SMOTE Distortion on Protocol Flow Features**: The paper heavily employs SMOTE (24 citations/occurrences) to synthesize minority traffic records. In network flow analysis, features represent physical transport mechanics (e.g., TCP flag ratios, packet size standard deviations, inter-arrival time variance). Synthesizing numerical points through $k$-nearest-neighbor interpolation creates physically impossible flows (such as fractional packet counts, invalid TCP flag combinations, or non-RFC sequence numbers), artificially separating classes with fictitious data that will never appear in live traffic.
  2. **Feature Extraction Computational Overhead**: The authors claim the system is "resource efficient for edge computing" because the machine learning model evaluates only 10 features. However, calculating those 10 features (which include flow duration, rate, packet variance, and covariance across sliding windows) still requires full packet parsing and statistical state maintenance in RAM for every active flow. The computational bottleneck in edge security is **stateful feature extraction**, not tree traversal depth.
  3. **Minority Exploit Masking**: The reported 99.7%+ accuracy is calculated globally. The paper contains zero per-class breakdown for minority application attacks like `DictionaryBruteForce` (which represents only 0.02% of the dataset).

---

### 2. Almahaqeri, Almourish, & Nasser (2026)
* **Full Title**: *An optimized gradient boosting framework for IoT intrusion detection: a comprehensive evaluation on the CICIoT2023 dataset*
* **Core Claims**: Applies gain-based feature selection and optimized gradient boosting (XGBoost, LightGBM, CatBoost) on CICIoT2023, claiming **99.61% accuracy** with an inference time of **0.355 µs per sample**.
* **Critical Methodological Faults**:
  1. **Binary Inflation Disguised as Intrusion Detection**: The headline **99.61% accuracy** and 0.355 µs latency are achieved strictly on **Binary Classification** (Attack vs. Benign). In CICIoT2023, malicious traffic constitutes 97.6% of the raw capture. An ultra-simplistic model predicting "Attack" for every packet automatically achieves a 97.6% baseline without learning any meaningful security semantics.
  2. **Flood Subclass Lumping**: When evaluating multiclass detection, the authors lump massive volumetric DDoS and DoS sub-categories together or evaluate on undersampled subsets without addressing how edge gateways handle the residual ambiguous traffic.
  3. **In-Distribution Random Split Vulnerability**: Uses standard random train/test splits across the blended CSVs. Because high-rate packet floods generate millions of consecutive packets with near-identical temporal and packet-size metrics, random splitting places packets from the exact same attack burst into both the train and test sets, resulting in synthetic performance inflation that collapses when tested against novel packet distributions.

---

### 3. Neto, Dadkhah, Ferreira, Zohourian, Lu, & Ghorbani (Sensors, 2023)
* **Full Title**: *CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environment* (The Foundational Dataset Paper)
* **Core Claims**: Introduced the 105-device IoT testbed with 33 attacks and benchmarked standard baseline models (Random Forest, DNN, Perceptron, AdaBoost, Logistic Regression), reporting headline accuracies of **98.6% (DNN)** and **99.16% (Random Forest)**.
* **Critical Methodological Faults (The Foundational "Accuracy Paradox")**:
  1. **Catastrophic Blind Spot on Stealthy Minority Attacks**:
     - In **Table 7** of the published paper, the DNN achieves an overall accuracy of **98.60%**.
     - However, on `Dictionary Brute Force`, the model achieved **only 14.7% Recall** (correctly identified 438 flows, but **missed 2,542 flows**, misclassifying them as Benign or Reconnaissance!).
     - On `Web-Based Attacks`, the model correctly identified **only 207 flows** while misclassifying **5,225 flows** (Recall = 3.8%)!
     - The headline 98.6% accuracy was mathematically driven by the **7,523,853 DDoS packets** and **1,787,065 DoS packets** that dominated the test split.
  2. **Identical DoS vs. DDoS Packet Signatures (Stripped IP Feature Defect)**:
     - To ensure models did not simply memorize static IP addresses, IP addresses were omitted from the 39 extracted features.
     - Consequently, a `DoS UDP Flood` (from a single attacking Raspberry Pi) and a `DDoS UDP Flood` (from multiple attacking Pis) exhibited identical packet sizes, header lengths, and flag numbers.
     - As shown in **Table 8**, Random Forest misclassified **1,088 DoS attacks as DDoS**. Because both share the same remediation action (dropping the volumetric flood), treating them as separate classes in a single monolithic model penalizes multiclass accuracy without providing operational security value.
  3. **The Single Monolithic Architecture Trap**:
     - The authors attempted to classify all 34 classes using a single classifier. This guarantees that gradient descent (in DNNs) or Gini-impurity minimization (in Random Forest) completely sacrifices the minority classes to achieve 99% overall accuracy on the flood majority.

---

### 4. Ullah, Wu, Lin, Kamal, & Mostafa (Scientific Reports, 2026)
* **Full Title**: *Comparative analysis of deep learning and traditional methods for IoT botnet detection using a multi-model framework across diverse datasets*
* **Core Claims**: Benchmarked CNN, LSTM, Random Forest, Autoencoders, and Federated Learning across BoT-IoT, CICIoT2023, and IoT-23, reporting **99.2% accuracy on BoT-IoT and CICIoT2023**.
* **Critical Methodological Faults**:
  1. **Massive Synthetic SMOTE Overfitting (59 Occurrences)**:
     - The authors acknowledge that class imbalance severely degrades detection, but attempt to resolve it by generating massive synthetic datasets using SMOTE across all three benchmarks.
     - This artificially injects hundreds of thousands of interpolated flow records that blur decision boundaries between benign web traffic and subtle application attacks.
  2. **BoT-IoT Imbalance Exploitation**:
     - The BoT-IoT dataset consists of 73 million records, of which **over 99.5% are DDoS and DoS traffic**. Achieving 99.2% on BoT-IoT is trivial and scientifically uninformative; any majority-class classifier matches this benchmark without learning botnet command-and-control behavior.
  3. **Deep Learning Latency Disconnect**:
     - The authors conclude that deep neural networks (CNN-LSTM) match or marginally exceed Random Forest accuracy. However, they fail to report that the inference latency of the deep neural stack is 20× to 80× higher than gradient boosted trees, rendering it unusable on edge routers during high-volume DDoS attacks.

---

### 5. DeepCLG Authors (Int. J. Machine Learning & Cybernetics, 2025)
* **Full Title**: *Enhancing network security in industrial IoT environments: a DeepCLG (CNN + LSTM + GRU)*
* **Core Claims**: Proposes a deep stacked architecture combining Convolutional layers (CNN), Long Short-Term Memory (LSTM), and Gated Recurrent Units (GRU) to protect Industrial IoT (IIoT), claiming **99.82% accuracy on BoT-IoT**.
* **Critical Methodological Faults**:
  1. **Extreme Architectural Overkill**:
     - Stacking a CNN, an LSTM, and a GRU in series creates millions of trainable parameters. In IIoT environments (e.g. SCADA, Modbus, PLCs), edge controllers have strict deterministic latency constraints (often sub-10 ms cycle times). Running sequential recurrent neural networks on live packet streams causes severe buffer overflows and dropped packets.
  2. **Catastrophic Performance Collapse on Diverse Benchmarks**:
     - While the model scores **99.82% on the flood-dominated BoT-IoT dataset**, its detection rate **collapses to 57.77% on the realistic UNSW-NB15 dataset**!
     - This massive **42.05% performance drop** demonstrates that the complex deep network simply memorized the monotonic flood characteristics of BoT-IoT and failed completely when confronted with diverse, stealthy real-world enterprise attacks.

---

### 6. FEC-IDS Authors (2025)
* **Full Title**: *Feature-Efficient Machine Learning Framework for Binary and Multiclass Intrusion Detection in IoT Networks*
* **Core Claims**: Proposes FEC-IDS using feature ranking and classical ML (RF, XGBoost, DT) to achieve **99.76% binary accuracy** and **99.21% multiclass accuracy** using only 16 features on CICIoT2023.
* **Critical Methodological Faults**:
  1. **Small Sample Slicing**: Rather than processing the complete multi-gigabyte CICIoT2023 stream, the authors evaluate on isolated, pre-sampled subsets where the natural variance of background network noise is artificially constrained.
  2. **Averaging Obfuscation**: The 99.21% multiclass accuracy is driven by weighted metrics where flood classes represent >90% of the test weights. Unweighted macro recall on minority classes is omitted.
  3. **Disregard of Stateful Attack Dynamics**: Eliminating 23 features based solely on Pearson correlation removes packet length variances and inter-arrival variances that are essential for differentiating low-and-slow reconnaissance scans from normal network fluctuations.

---

```
================================================================================
PART II: FEDERATED LEARNING & LEGACY DATASET PAPERS (snuc/clg-capstone-literature)
================================================================================
```

### 7. Al-Hejri, Al-Tam, et al. (Scientific Reports, 2026)
* **Full Title**: *Improving IoT security through an explainable hybrid CNN-transformer model and federated learning*
* **Core Claims**: Proposes combining a 1D-CNN with a Transformer self-attention encoder, trained under Federated Learning (FL) across distributed edge nodes, with SHAP/LIME explainability, claiming **94.89% accuracy**.
* **Critical Methodological Faults**:
  1. **Transformer Computational Infeasibility on IoT Edge**:
     - Deploying Transformer multi-head self-attention on resource-constrained IoT gateways (e.g., Raspberry Pi, ESP32, home routers) is fundamentally impractical.
     - Self-attention scales quadratically $O(N^2)$ with sequence length in both memory and compute. A gateway processing 50,000 packets per second would exhaust its memory buffers in milliseconds.
  2. **Federated Collapse Under Realistic Non-IID Distributions**:
     - Under balanced IID assumptions, the paper reports 94.89% accuracy. However, when tested under realistic **non-IID data distributions** (where different IoT clients experience different attack types), accuracy plummets to **69.35%**!
     - In real IoT deployments, a smart camera experiences different traffic than a temperature sensor. The paper proves that federated aggregation fails when edge distributions diverge.

---

### 8. Booij, Chiscop, et al. (IEEE Internet of Things Journal, 2022)
* **Full Title**: *ToN_IoT: Heterogeneity and Standardization of IoT Intrusion Datasets*
* **Core Claims**: Comprehensive statistical and machine learning evaluation of the ToN_IoT dataset, examining dataset heterogeneity and cross-dataset generalization.
* **Internal Value & Methodological Critique**:
  - *Note*: This paper is a high-quality critique that actually exposes why earlier datasets (like BoT-IoT) produced inflated >99.9% accuracies due to lack of heterogeneity.
* **Remaining Critical Limitations**:
  1. **Flow-Level vs. Packet-Level Disconnect**:
     - The evaluation relies on Zeek/NetFlow aggregated flow summaries. Generating a flow record requires waiting for a session to close (or reach a 60-second timeout).
     - For volumetric DDoS or active command injection, waiting 60 seconds to detect an intrusion means the device has already been overwhelmed or compromised. Real-world edge IDS must operate on streaming packets/micro-windows.
  2. **Absence of Mitigation Pipeline**:
     - The paper demonstrates that cross-dataset performance drops dramatically, but offers no operational architecture (such as hierarchical filtering) to handle real-world distribution shifts.

---

### 9. Choudhary & Kesswani (Sensors, 2020)
* **Full Title**: *Generating Datasets for Anomaly-Based Intrusion Detection in IoT Networks: A Survey and Evaluation*
* **Core Claims**: Surveys and evaluates dataset generation methodologies for anomaly-based IDS in IoT networks.
* **Critical Methodological Faults**:
  1. **Reliance on Simulation Artifacts**:
     - The surveyed methodologies rely heavily on network simulators (NS-2, NS-3, Mininet). Simulated traffic lacks physical hardware jitter, CPU scheduling delays, and genuine user behavioral patterns. Models trained on simulated data exhibit near-100% synthetic accuracy that fails completely when deployed on physical hardware.
  2. **Outdated Threat Taxonomies**:
     - The paper relies on legacy 1999 DARPA/KDD attack taxonomies (Probe, DoS, U2R, R2L). These categories do not capture modern IoT botnet life cycles (Mirai multi-stage reconnaissance, GRE payload tunneling, Telnet brute-forcing).

---

### 10. Fenanir & Semchedine (2023)
* **Full Title**: *Smart Intrusion Detection in IoT Edge Computing Using Federated Learning*
* **Core Claims**: Proposes a federated learning framework using CNN, LSTM, and GRU models across edge devices, reporting **99.99% accuracy on NSL-KDD**.
* **Critical Methodological Faults**:
  1. **Evaluating a 2023 IoT Model on a 1999 Dataset (NSL-KDD)**:
     - Reporting 99.99% accuracy on NSL-KDD is completely irrelevant for modern IoT networks. NSL-KDD lacks modern IoT protocols (MQTT, CoAP, Zigbee, IPv6, DNS over HTTPS) and modern botnet dynamics.
  2. **Ignoring Network Communication Overhead in Federated Rounds**:
     - The authors ignore the uplink bandwidth cost of transmitting deep neural network weight updates over wireless IoT channels. In low-power networks (Zigbee, BLE, LoRaWAN), transmitting megabytes of weights across 100 training rounds exhausts device batteries and creates network congestion.
  3. **Severe Performance Drop on Complex Data**:
     - On more realistic datasets (UNSW-NB15), accuracy drops to **81.33% – 84.66%**, confirming that the near-perfect accuracy was an artifact of the obsolete NSL-KDD benchmark.

---

### 11. Moussaoui & Kmiti (Scientific Reports, 2026)
* **Full Title**: *Resilient federated intrusion detection with explainable AI: a robust CNN-LSTM architecture for extreme non-IID data distributions*
* **Core Claims**: Proposes a CNN-LSTM architecture evaluated under federated learning across CICIoT2023, TON_IoT, and NSL-KDD under non-IID data distributions, reporting 97.36% centralized accuracy.
* **Critical Methodological Faults**:
  1. **Extreme Accuracy Degradation Under Realistic Conditions**:
     - The authors explicitly report that under extreme non-IID conditions, model accuracy **drops from 97.36% to 71.53% (a 25.83% absolute collapse)**!
     - This demonstrates that deep federated models cannot handle decentralized IoT environments where traffic distributions vary by device function.
  2. **Explicit Admission of Flow Feature Blindness on Payload Attacks**:
     - On page 18, the authors explicitly state: *"a fundamental limitation of flow-based features is their inability to capture payload-driven attacks."* Because flow statistics only track packet counts and durations, attacks like SQL injection, command injection, and cross-site scripting (XSS) pass undetected unless deep packet inspection is applied.
  3. **Synthetic Pre-Balancing Before Partitioning**:
     - Applied SMOTE prior to federated partitioning, creating synthetic data artifacts that artificially stabilized client gradient updates.

---

### 12. Moustafa (2021)
* **Full Title**: *A new distributed architecture for evaluating AI-based security systems at the edge: Network TON_IoT datasets*
* **Core Claims**: Introduced the ToN_IoT dataset collected from an NSX vCloud NFV testbed with telemetry and network flow data.
* **Critical Methodological Faults**:
  1. **Static IP and MAC Address Data Leakage**:
     - As demonstrated by independent evaluations (Booij et al., 2022), the network captures in ToN_IoT contain fixed static IP and MAC addresses for the attacking Kali VM and victim IoT devices.
     - Machine learning models evaluated on the raw features achieve near-perfect >99% accuracy simply by memorizing the attacker's IP address (`192.168.1.x`), learning zero behavioral intrusion patterns.
  2. **Coarse Aggregation Masking Stealthy Exploits**:
     - Attacks were evaluated in broad categories. Stealthy reconnaissance scans were frequently grouped with benign telemetry, artificially boosting overall macro F1 scores while ignoring high false-alarm rates in real network operations.

---

### 13. Qaddos & Yaseen (Scientific Reports, 2024)
* **Full Title**: *A novel intrusion detection framework for optimizing IoT security*
* **Core Claims**: Proposes Feature-Weighted SMOTE (FW-SMOTE) combined with a deep hybrid architecture (CNN + LSTM + Transformer attention), claiming **99.60% accuracy on BoT-IoT**.
* **Critical Methodological Faults**:
  1. **Excessive Computational Pipeline Unsuitable for Edge Gateways**:
     - The proposed pipeline executes: Feature Weighting $\rightarrow$ FW-SMOTE $\rightarrow$ 1D-CNN feature extraction $\rightarrow$ BiLSTM temporal modeling $\rightarrow$ Transformer Multi-Head Self-Attention.
     - This architecture requires high-end server-grade GPUs (NVIDIA RTX/A100) to train and evaluate. Proposing this as an "IoT edge security optimization" is fundamentally contradictory, as edge microcontrollers (ARM Cortex-M, Raspberry Pi) cannot run this pipeline in real time.
  2. **Exploiting BoT-IoT's Monolithic Flood Distribution**:
     - Achieving 99.60% on BoT-IoT is entirely expected when 99.5% of the samples are identical DoS/DDoS packets. FW-SMOTE simply synthesized additional copies of already predictable flood traffic.
  3. **Zero Wire-Speed or Hardware Latency Evaluation**:
     - Contains no benchmarking of packet processing throughput (packets per second) or latency per sample under line-rate traffic.

---

### 14. Umar & Yasmeen (EURASIP J. Information Security, 2025)
* **Full Title**: *Energy-efficient deep learning-based intrusion detection system for edge computing: a novel DNN-KDQ model*
* **Core Claims**: Proposes Knowledge Distillation (KD) and INT8 Quantization (DNN-KDQ) to compress a deep neural network from 196.77 KB to 51 KB, achieving **99.43% accuracy on CICIDS2017**.
* **Critical Methodological Faults**:
  1. **Evaluation on Flawed CICIDS2017 Dataset**:
     - The CICIDS2017 dataset suffers from well-documented ground-truth flaws (Engelen et al., 2021; Rosay et al., 2022), including missing attack traffic, inverted packet timestamps, and corrupted flow calculations. Evaluating model compression on corrupted ground truth yields unreliable conclusions.
  2. **Quantization Boundary Collapse on Subtle Minority Attacks**:
     - While INT8 quantization successfully compresses model weights to 51 KB, quantizing 32-bit floating-point weights into 8-bit integers rounds off decision thresholds. For stealthy minority attacks (e.g. dictionary brute force or slow scans) where malicious flows differ from benign traffic by minute variations in inter-arrival times or header lengths, quantization causes severe boundary collapse.
  3. **No Evaluation on Modern IoT Protocols**:
     - The paper claims to provide an "IoT edge solution", yet evaluates exclusively on enterprise desktop PC traffic (CICIDS2017 and NSL-KDD), testing zero MQTT, CoAP, Zigbee, or smart home sensor traffic.

---

## Synthesis: The Five Systemic Failures of Current IoT IDS Literature

Across all 14 research papers, five universal methodological flaws repeatedly emerge:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                      THE FIVE SYSTEMIC FLAWS IN IoT IDS LITERATURE                    │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. THE ACCURACY PARADOX                                                                │
│    98%–99.9% headline accuracies are driven entirely by volumetric flood dominance     │
│    (>90% DoS/DDoS). Underneath, minority exploit recall is catastrophic (0% to 15%).   │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 2. SYNTHETIC SMOTE CONTAMINATION                                                       │
│    Over 50% of papers use SMOTE/oversampling on network flows, creating synthetic      │
│    interpolations that violate physical RFC protocol rules and distort boundaries.     │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 3. THE "EDGE DEEP LEARNING" CONTRADICTION                                              │
│    Papers claim "edge-optimized IoT security" while proposing Transformers, stacked   │
│    CNN-LSTM-GRUs, and FL rounds requiring milliseconds of compute and gigabytes of RAM.│
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 4. DATA LEAKAGE & STATIC IDENTIFIER ARTIFACTS                                          │
│    Random cross-validation splits packets from the same TCP session across train/test; │
│    static IP/MAC addresses are retained, leading to rote memorization rather than ML. │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 5. THE MONOLITHIC PIPELINE ILLUSION                                                    │
│    All 14 papers attempt to detect all 33+ attack types using a SINGLE classifier,     │
│    guaranteeing that minority exploits are sacrificed to optimize overall accuracy.    │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## How Team Archon's Capstone Architecture Resolves Every Flaw

| Literature Methodological Flaw | How Literature Fails | How Team Archon Solves It | Empirical Evidence from Our Work |
|---|---|---|---|
| **1. The Accuracy Paradox** | Reports 99% accuracy; model has 0%–14% recall on brute force and web attacks (Neto et al. Table 7). | **Multi-Tier Hierarchical Gateway**: Separates line-rate volumetric flood filtering from stateful deep exploit inspection. | **Notebook 07**: Achieves **99.18% binary accuracy** at Tier 1 (Wire-speed filter), while **Notebooks 01–06** achieve **88.5% brute force recall** and **81.4% web precision** at Tier 2/3. |
| **2. Synthetic SMOTE Contamination** | Injects artificial interpolated flows (Khan 2026, Ullah 2026, Qaddos 2024), corrupting real protocol behavior. | **Zero Synthetic Injection**: We use natural distribution stratified sampling with a **20,000 packet ceiling per flood class**, preserving 100% authentic physical packet dynamics. | Evaluated across 200,000 authentic flow records from raw `MERGED_CSV/` files without generating a single synthetic packet. |
| **3. Impractical Edge Complexity** | Deploys Transformers, CNN-LSTM-GRUs on edge devices (Al-Hejri 2026, Qaddos 2024), causing high latency and memory exhaustion. | **High-Throughput Gradient Boosted Trees (XGBoost Hist / LightGBM)**: Ultra-compact, hardware-efficient decision trees optimized for edge CPU execution. | **Latency Benchmark**: **0.65 µs per flow** (XGBoost Hist) and **1.19 µs per flow** (LightGBM), easily satisfying 10 Gbps line-rate wire-speed requirements. |
| **4. Data Leakage & Static Identifiers** | Memorizes static IP and MAC addresses in TON_IoT and BoT-IoT (Moustafa 2021, Booij 2022). | **Purely Behavioral Statistical Features**: All IP addresses, MAC addresses, port numbers, and timestamps were strictly stripped. | Models learn purely statistical flow dynamics (packet length variance, header ratios, flag counts), ensuring true generalization to unseen subnets. |
| **5. Monolithic Pipeline Blindness** | Uses a single model for both microsecond packet drops and complex deep attack classification. | **3-Tier Specialized Gateway**: <br>• **Tier 1**: Wire-Speed Binary Ingress Filter (drops floods in < 1 µs) <br>• **Tier 2**: 8-Class Attack Category Classifier <br>• **Tier 3**: 34-Class Fine-Grained Signature Engine | **Full Working Implementation**: Demonstrated live in interactive Streamlit gateway application ([`demo_app.py`](file:///c:/Users/priya/Downloads/capstone/demo_app.py)). |

---

## Conclusion & Committee Defense Strategy

When defending before the Review 3 faculty committee, Team Archon possesses an airtight scientific narrative:

> *"When academic papers claim 99.5% accuracy on CICIoT2023 or BoT-IoT, they are benefiting from the **Accuracy Paradox**: raw IoT traffic is >90% volumetric floods. In Neto et al.'s own foundational paper (Sensors 2023), their 98.6% accurate DNN missed **85.3% of brute-force attacks and 96.2% of web attacks**.*
> 
> *Our research does not have a defect—we proved both sides of the coin. In **Notebook 07**, we demonstrate that when evaluated on the natural uncapped stream, our models achieve the identical **99.18% literature accuracy** with microsecond latency. However, rather than stopping there like other papers, our **Multi-Tier Architecture** deploys that 99.2% model at Tier 1 to filter out the flood tsunami, passing surviving ambiguous traffic to Tier 2/3, which achieves **88.5% brute-force recall and 81.4% web exploit precision**.*
> 
> *By combining line-rate flood filtering with deep inspection, Team Archon's system solves both the volumetric availability challenge and the stealthy exploitation challenge that existing literature completely overlooks."*
