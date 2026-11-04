# Stage 7: Literature Review and Research Gap Analysis

**Project Title**: Empirical Evaluation and Methodological Auditing of Machine Learning Baselines for IoT Network Intrusion Detection on the CICIoT2023 Benchmark  
**Deliverable**: Comprehensive Literature Review, Methodological Audit, and Research Gap Synthesis  
**Repository References**: 
- Research Archive: `research/` (19 PDF documents: 14 distinct peer-reviewed studies, 1 duplicate copy, 1 dataset datasheet, 3 capstone technical analyses)
- Stage 6 Synthesis: [`results_v2/STAGE_6_RESEARCH_SYNTHESIS.md`](STAGE_6_RESEARCH_SYNTHESIS.md)
- Stage 4 Evaluation: [`results_v2/STAGE_4_EVALUATION_REPORT.md`](STAGE_4_EVALUATION_REPORT.md)
- Stage 5 Error Analysis: [`results_v2/STAGE_5_ERROR_ANALYSIS.md`](STAGE_5_ERROR_ANALYSIS.md)
- Centralized Ledger: [`results_v2/tables/centralized_experiment_ledger.csv`](tables/centralized_experiment_ledger.csv)
- Master Test Results: [`results_v2/tables/stage4_test_results.csv`](tables/stage4_test_results.csv)

---

## 1. Executive Summary & Review Scope

Modern Network Intrusion Detection Systems (NIDS) designed for the Internet of Things (IoT) operate under severe real-world operational constraints: high packet arrival rates, constrained compute and memory headroom on edge gateways, heterogeneous communication protocols, and extreme natural class imbalance. In recent cybersecurity literature evaluating contemporary IoT benchmarks—most notably **CICIoT2023**, **ToN_IoT**, and **BoT-IoT**—published studies routinely report headline classification accuracies exceeding **98.5% to 99.9%**.

However, a systematic methodological examination of the peer-reviewed literature reveals that these impressive headline figures frequently mask critical methodological vulnerabilities, dataset-induced artifacts, and severe operational blind spots:
1. **The Accuracy Paradox under Severe Class Imbalance**: High headline accuracy is largely driven by volumetric flood dominance (DDoS and DoS account for $>85\%$ of raw traffic), while detection recall on stealthy, low-footprint application attacks (e.g., dictionary brute-force, web vulnerabilities, SQL injection, and command backdoors) drops to near-zero levels.
2. **Synthetic Data Contamination via SMOTE**: Rather than addressing imbalance through cost-sensitive loss formulations or natural stratified sampling, over half of surveyed studies apply $k$-nearest-neighbor interpolation (SMOTE) to continuous and discrete network flow features, generating physically invalid protocol states that distort decision boundaries.
3. **Data Leakage via Unchecked Feature Duplication**: Pervasive feature vector redundancy and identical-feature records across capture sessions are routinely ignored. When standard random train-test splits (e.g., 80/20 random holdouts or $k$-fold cross-validation) are applied to datasets with high feature redundancy, identical feature vectors are placed in both training and test partitions, inflating reported performance.
4. **Neglect of Conflicting Label Collisions**: Thousands of network flows share identical feature vectors while carrying contradictory ground-truth labels (e.g., identical flow features assigned to both `BENIGN` and `DOS-TCP_FLOOD`). None of the reviewed studies investigate or isolate this feature-label ambiguity.
5. **Absence of Temporal Cross-Capture Evaluation**: Evaluations are conducted almost exclusively on in-distribution random splits sampled from the same capture sessions. No prior study has systematically evaluated model stability across distinct, subsequent capture sessions on the same testbed.
6. **The "Edge Deep Learning" Latency Contradiction**: Several studies propose deep recurrent, convolutional, or Transformer architectures with multi-head self-attention for "edge IoT security," despite these models incurring millisecond-level inference latencies and severe memory footprints that fail line-rate throughput requirements.

This report systematically reviews all **19 documents** in the repository's `research/` archive, extracts their exact experimental parameters, evaluates their handling of methodological vulnerabilities, compares them against our 21-model benchmark, and defines the precise research gap addressed by our investigation.

---

## 2. Master Comparison Matrix of Reviewed Literature

The table below summarizes the 14 distinct peer-reviewed research papers analyzed from `research/`.

| # | Citation & Venue | Year | Datasets Evaluated | Evaluated CICIoT2023? | Proposed Architecture / Models | Reported Metrics | Train/Test Split Methodology | Handled Duplicates / Collisions? | Temporal / Drift Evaluation? |
|---|---|---|---|:---:|---|---|---|:---:|:---:|
| **1** | **Khan et al.**<br>*Sci. Rep.* (Nature) [1] | 2026 | CICIoT2023, TON_IoT, IoT-23 | **Yes** | BGL-RID (Boruta + GFS + LightGBM / CatBoost) | Acc: 99.72%–99.94% | 80/20 train/test holdout; scaler on train | Feature pruning; no vector collision audit | Timestamps stripped; no cross-session test |
| **2** | **Almahaqeri et al.**<br>*Preprint / Review* [2] | 2026 | CICIoT2023 | **Yes** | GFS + XGBoost, LightGBM, CatBoost | Acc: 99.61% (Binary); Lat: 0.355 $\mu$s | 5-fold CV & 80/20 holdout; scaler on train | Feature pruning; no vector collision audit | Discusses drift concept; no temporal test |
| **3** | **Neto et al.**<br>*Sensors* (MDPI) [3] | 2023 | CICIoT2023 (Foundational) | **Yes** | RF, DNN, Perceptron, AdaBoost, LR, DT, NB, KNN | Acc: 98.60% (DNN), 99.16% (RF) | 80/20 random stratified split | Stripped IPs; no duplicate/collision audit | Single-stream capture; no cross-session test |
| **4** | **Ullah et al.**<br>*Sci. Rep.* (Nature) [4] | 2026 | BoT-IoT, CICIoT2023, IoT-23 | **Yes** | Multi-Model (CNN, LSTM, RF, AE, FedAvg) | Acc: 99.20% (BoT/CIC), 91.5% (IoT-23) | Random split; SMOTE on training data | Removed duplicate full rows; no collision audit | Recurrent LSTM; no cross-session test |
| **5** | **Gulzar & Mustafa**<br>*Int. J. ML & Cyber.* [5] | 2025 | TON_IoT, BoT-IoT, NSL-KDD, UNSW-NB15 | No (mentions) | DeepCLG (1D-CNN + LSTM + GRU Stack) | Acc: 99.82% (BoT-IoT), 57.77% (UNSW) | 80/20 random train/test split | Feature selection; no collision audit | Recurrent temporal units; no cross-session test |
| **6** | **Abdelrahman et al.**<br>*IEEE ICEENG / IICC* [6] | 2026 | CICIoT2023 | **Yes** | FEC-IDS (Feature Ranking + RF, XGBoost, DT) | Acc: 99.76% (Bin), 99.21% (Cat), 98.88% (Fine) | Random holdout on sampled slices | Removed duplicate full rows; no collision audit | No temporal or cross-session test |
| **7** | **Al-Hejri et al.**<br>*Sci. Rep.* (Nature) [7] | 2026 | TON_IoT, BoT-IoT, IoT-23, NSL-KDD | No | Hybrid 1D-CNN-Transformer + FL + XAI | Acc: 94.89% (IID), 69.35% (Non-IID) | Federated client partitioning (IID vs non-IID) | Feature selection; no collision audit | Self-attention temporal modeling; no drift test |
| **8** | **Booij et al.**<br>*IEEE Internet Things J.* [8] | 2022 | TON_IoT, BoT-IoT, UNSW-NB15, NSL-KDD | No | Random Forest, Decision Tree | Acc: 99.68% (in-domain); low cross-dataset | In-domain 70/30 split; cross-dataset eval | Evaluated feature distributions; noted static IP leak | Evaluated cross-dataset drift; no temporal split |
| **9** | **Essop et al.**<br>*Sensors* (MDPI) [9] | 2021 | Cooja/Contiki IoT Simulations | No | Dataset Generation & ML Baselines | Survey & Empirical Dataset Analysis | Simulation-based scenario generation | Analyzed synthetic artifacts; no collision audit | Noted lack of behavioral temporal drift |
| **10** | **Fenanir & Semchedine**<br>*Rev. Intell. Artif.* [10] | 2023 | NSL-KDD, IoT-23, UNSW-NB15 | No | Federated Deep Learning (CNN, LSTM, GRU) | Acc: 99.99% (NSL-KDD), 81.33% (UNSW) | Federated client split with local test sets | Preprocessing for missing values; no collision audit | Recurrent memory; no cross-session test |
| **11** | **Moussaoui et al.**<br>*Sci. Rep.* (Nature) [11] | 2026 | CICIoT2023, TON_IoT, NSL-KDD | **Yes** | CNN-LSTM + Federated Learning + XAI | F1: 97.36% (Centr.), 71.53% (Dirichlet $\alpha=0.01$) | Dirichlet Non-IID client split; scaler on train | Stripped redundant features; no collision audit | Noted flow feature blindness on payloads |
| **12** | **Moustafa**<br>*SCS / Tech Report* [12] | 2021 | Network TON_IoT Datasets | No | Distributed Edge AI (RF, LSTM) | Acc: ~98.5% across edge nodes | Standard train/test partitions | Documented raw features; static IPs leaked | Network summary windows; no cross-session test |
| **13** | **Qaddos et al.**<br>*Sci. Rep.* (Nature) [13] | 2024 | BoT-IoT, IoT-23, NSL-KDD, UNSW-NB15 | No | FW-SMOTE + 1D-CNN + BiLSTM + Attention | Acc: 99.60% (BoT-IoT), 99.16% (UNSW) | 80/20 train/test split; FW-SMOTE on train | Pruned redundant features; no collision audit | BiLSTM sequence modeling; no cross-session test |
| **14** | **Umar et al.**<br>*J. Cloud Comput.* [14] | 2025 | CICIDS2017, NSL-KDD | No | DNN-KDQ (Knowledge Distillation + INT8 Quant.) | Acc: 99.43% test accuracy; 51 KB footprint | Standard holdout split; teacher-student train | Preserved weights; no collision audit | Inference latency profiling; no cross-session test |

*Note on Repository Files*: 
- File 15 (`Khan_Alam_ResourceEfficient...pdf`) is an identical copy of File 1 (`A resource efficient...pdf`).
- File 10 (`Dataset.pdf`) is the 2-page publication flyer for CICIoT2023 by Neto et al.
- File 8 (`ChoudharyTanwar_...pdf`) is the published paper by Essop et al. (Sensors 2021); the file was named after survey citations in early project phases.
- Files 6, 7, and 14 (`CICIoT2023 Capstone Research Analysis...`, `CICIoT2023 Model Pipeline Analysis...`, and `IoT Edge IDS Optimization Research Original...`) are internal capstone literature reviews and technical blueprints documenting historical background and edge deployment objectives.

---

## 3. Systematic In-Depth Review of Surveyed Studies

### 3.1 Studies Directly Benchmarking the CICIoT2023 Benchmark

#### 1. Neto et al. (Sensors, 2023) — The Foundational CICIoT2023 Benchmark
- **Bibliographic Details**: E. C. P. Neto, S. Dadkhah, R. Ferreira, A. Zohourian, R. Lu, and A. A. Ghorbani, "CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environment," *Sensors*, vol. 23, no. 13, art. 5941, 2023. DOI: `10.3390/s23135941`.
- **Dataset Evaluated**: Introduced the CICIoT2023 dataset generated on a physical testbed of 105 IoT devices executing 33 attacks categorized into 7 attack categories plus Benign (34 classes total). Extracted 46 raw features (39 numerical flow features plus metadata/labels).
- **Preprocessing & Splitting**: Evaluated on an 80/20 random stratified split across the blended dataset. Continuous features were normalized.
- **Models & Classification Tasks**: Benchmarked 8 baseline algorithms: Random Forest, Decision Tree, Logistic Regression, Perceptron, Deep Neural Network (DNN), AdaBoost, Naive Bayes, and K-Nearest Neighbors across 3 granularities: 34-class fine-grained, 8-class category, and 2-class binary.
- **Reported Metrics**: Reported overall Accuracy, Precision, Recall, and F1-score (e.g., DNN: 98.60% accuracy, Random Forest: 99.16% accuracy in multiclass).
- **Audit of Methodological Factors**:
  - *Duplicates and Collisions*: The authors explicitly removed IP addresses and hardware MAC identifiers to prevent rote memorization of attacking nodes. However, they did not audit feature-vector duplication or label collisions among the remaining 39 behavioral features.
  - *Class Imbalance & Minority Starvation*: While the paper reported impressive $>98\%$ headline accuracy, Table 7 reveals severe minority attack starvation in the DNN baseline: `DictionaryBruteForce` achieved only **14.7% Recall** (correctly identified 438 flows, missed 2,542 flows); `Web-Based Attacks` achieved only **3.8% Recall** (correctly identified 207 flows, missed 5,225 flows). Aggregate accuracy was dominated by 7.5 million DDoS and 1.7 million DoS packets.
  - *Temporal Evaluation*: Evaluated purely on in-distribution random splits of the blended capture stream; no cross-session or temporal stability evaluation was conducted.
- **Limitations Relevant to Our Work**: Establishes the foundational "accuracy paradox" of CICIoT2023, where high aggregate metrics mask near-complete failure on stealthy minority intrusions. Omitting source IP entropy also produced structural confusion between DoS and DDoS packet floods.

#### 2. Khan & Alam et al. (Scientific Reports, 2026) — Hybrid Feature Selection for Edge Computing
- **Bibliographic Details**: M. Z. Khan, M. Alam, I. Alam, M. U. Bokhari, Z. Ashraf, M. Z. Khan, and F. S. Masoodi, "A resource efficient IoT intrusion detection model using hybrid feature selection for edge computing," *Scientific Reports*, vol. 16, art. 24187, 2026. DOI: `10.1038/s41598-026-59156-9`.
- **Dataset Evaluated**: CICIoT2023, TON_IoT, and IoT-23.
- **Preprocessing & Splitting**: Standard 80/20 holdout partition. Scaling parameters and feature selection (Boruta + Greedy Forward Selection) were fitted exclusively on the training partition and evaluated on the test set.
- **Models & Classification Tasks**: Evaluated LightGBM, CatBoost, and XGBoost classifiers on reduced feature subsets (pruned down to 10–15 features).
- **Reported Metrics**: Reported 99.72% to 99.94% overall classification accuracy across datasets, along with a custom "efficiency ratio" relating accuracy to training time and memory usage.
- **Audit of Methodological Factors**:
  - *Class Imbalance & SMOTE*: The authors extensively utilized SMOTE (17 mentions) to generate synthetic samples for minority attack classes. In network flow analysis, features represent physical packet dynamics (e.g., TCP flag ratios, packet size standard deviations, inter-arrival time variance). Synthesizing numerical points through $k$-NN linear interpolation creates physically impossible flows (e.g., fractional packet counts or non-RFC flag states) that will never appear in live network traffic.
  - *Duplicates and Collisions*: Pruned redundant features but did not audit identical feature vectors or conflicting label collisions across splits.
  - *Temporal Evaluation*: Stripped timestamps; no temporal cross-capture validation was performed.
- **Limitations Relevant to Our Work**: Claims edge efficiency based on evaluating 10–15 features in the ML classifier. However, extracting those features still requires maintaining full stateful packet parsing in RAM for every active flow. Furthermore, synthetic SMOTE oversampling distorts the authentic protocol distribution.

#### 3. Almahaqeri et al. (2026) — Comprehensive Gradient Boosting on CICIoT2023
- **Bibliographic Details**: S. A. Almahaqeri, M. H. Almourish, A. A. Nasser, A. S. A. Alghawli, A. A. K. Elsayed, and A. N. Alhejoj, "An optimized gradient boosting framework for IoT intrusion detection: a comprehensive evaluation on the CICIoT2023 dataset," Preprint / Manuscript Under Review, 2026.
- **Dataset Evaluated**: CICIoT2023 exclusively.
- **Preprocessing & Splitting**: Chunked ingestion, stratified undersampling, and leakage-safe standard scaling ($\mu, \sigma$ fitted exclusively on training data). Evaluated using 5-fold cross-validation and held-out test evaluation.
- **Models & Classification Tasks**: XGBoost, LightGBM, and CatBoost evaluated across 3 tasks: binary detection, 8-class attack family classification, and 34-class fine-grained attribution.
- **Reported Metrics**: Reported 99.61% accuracy and 0.355 $\mu$s per-sample inference latency for binary classification.
- **Audit of Methodological Factors**:
  - *Binary Inflation*: The headline 99.61% accuracy was achieved strictly on the binary task. Because malicious traffic constitutes over 97% of the raw capture, predicting the majority class provides a 97%+ baseline without learning nuanced threat semantics.
  - *Duplicates and Collisions*: Applied gain-based feature selection to remove uninformative flow features, but did not check for identical feature vectors sharing conflicting labels.
  - *Temporal Evaluation*: Discusses concept drift conceptually, but all evaluations were conducted on in-distribution random splits of the same dataset.
- **Limitations Relevant to Our Work**: Evaluated on pre-sampled, undersampled subsets that constrain background noise variance; did not audit label ambiguity or evaluate across temporal capture sessions.

#### 4. Abdelrahman et al. / FEC-IDS (IEEE ICEENG / IICC, 2026) — Feature-Efficient ML Framework
- **Bibliographic Details**: A. Abdelrahman, A. Elhossieny, I. Bassiony, and A. Mattar, "Feature-Efficient Machine Learning Framework for Binary and Multiclass Intrusion Detection in IoT Networks," *2026 ICEENG International Conference for Innovations in Intelligent Computing and Cybersecurity (IICC)*, IEEE, 2026. DOI: `10.1109/IICC69623.2026.11582694`.
- **Dataset Evaluated**: CICIoT2023.
- **Preprocessing & Splitting**: Evaluated on pre-sampled slices with feature normalization. Random holdout partitions.
- **Models & Classification Tasks**: Random Forest, XGBoost, and Decision Tree across binary (2 classes), multiclass category (8 classes), and fine-grained (34 classes).
- **Reported Metrics**: 99.76% binary accuracy using 7 features; 99.21% category accuracy using 16 features; 98.88% fine-grained accuracy (75.0% F1).
- **Audit of Methodological Factors**:
  - *Duplicates and Collisions*: Preprocessing removed "duplicate records" (exact duplicate full rows), but did not audit identical feature vectors mapping to conflicting ground-truth labels.
  - *Class Imbalance*: Evaluated on small sampled subsets; reports weighted macro averages that allow dominant flood classes to obscure low recall on stealthy attack vectors.
  - *Temporal Evaluation*: No cross-capture or temporal evaluation was performed.
- **Limitations Relevant to Our Work**: Relies on small sampled slices that artificially constrain network noise variance; feature reduction prunes variance indicators critical for low-and-slow reconnaissance.

#### 5. Ullah et al. (Scientific Reports, 2026) — Multi-Model Framework Across Diverse Datasets
- **Bibliographic Details**: I. Ullah, N.-C. Wu, C.-F. Lin, M. S. Kamal, and S. M. Mostafa, "Comparative analysis of deep learning and traditional methods for IoT botnet detection using a multi-model framework across diverse datasets," *Scientific Reports*, vol. 16, art. 24190, 2026. DOI: `10.1038/s41598-026-59142-1`.
- **Dataset Evaluated**: BoT-IoT, CICIoT2023, and IoT-23.
- **Preprocessing & Splitting**: Removed exact duplicate records; applied SMOTE oversampling to rebalance training data.
- **Models & Classification Tasks**: Evaluated a broad multi-model framework: 1D-CNN, LSTM, Random Forest, Autoencoder, and Federated Learning (FedAvg) for botnet detection.
- **Reported Metrics**: 99.20% accuracy on BoT-IoT and CICIoT2023; 91.50% on IoT-23.
- **Audit of Methodological Factors**:
  - *Synthetic SMOTE Distortion*: The paper acknowledges severe class imbalance, but attempts to resolve it via aggressive synthetic oversampling (59 mentions of SMOTE). This injects hundreds of thousands of interpolated flow vectors that blur the natural decision boundary between benign background traffic and stealthy attacks.
  - *BoT-IoT Imbalance*: Relies heavily on BoT-IoT, which consists of $>99.5\%$ DoS/DDoS traffic, rendering high accuracy trivial to achieve.
  - *Inference Latency*: Deep recurrent models (CNN-LSTM) matched or marginally exceeded Random Forest accuracy, but incurred 20× to 80× higher inference latency, making them impractical for wire-speed edge deployment during packet floods.
  - *Collisions & Temporal Stability*: No collision analysis or cross-session temporal validation was performed.

#### 6. Moussaoui et al. (Scientific Reports, 2026) — Resilient Federated IDS under Extreme Non-IID
- **Bibliographic Details**: J.-E. Moussaoui, M. Kmiti, Y. Maleh, and K. El Gholami, "Resilient federated intrusion detection with explainable AI: a robust CNN-LSTM architecture for extreme non-IID data distributions," *Scientific Reports*, vol. 16, art. 32367, 2026. DOI: `10.1038/s41598-026-61994-6`.
- **Dataset Evaluated**: CICIoT2023, TON_IoT, and NSL-KDD.
- **Preprocessing & Splitting**: Dirichlet-based non-IID client partitioning ($\alpha \in \{0.01, 0.1, 0.5, 5.0, 50.0\}$). Scaling parameters computed strictly on training data.
- **Models & Classification Tasks**: Hybrid CNN-LSTM architecture evaluated under federated learning (FedAvg) and centralized baselines.
- **Reported Metrics**: 97.36% centralized F1-score; however, under extreme non-IID skew ($\alpha = 0.01$), performance collapses to 71.53% (a 25.83% absolute drop).
- **Audit of Methodological Factors**:
  - *Payload Blindness of Flow Features*: The authors explicitly note a fundamental limitation of statistical flow features: *"a fundamental limitation of flow-based features is their inability to capture payload-driven attacks."* Because flow statistics track packet headers and timings, web attacks (e.g., SQLi, XSS, Command Injection) that match standard HTTP request sizes pass undetected without deep packet inspection.
  - *SMOTE Prior to Partitioning*: Applied SMOTE before federated client partitioning, injecting synthetic data into client updates.
  - *Collisions & Temporal Evaluation*: No feature-hash collision audit or cross-session temporal evaluation was conducted.

---

### 3.2 Studies Evaluating Other IoT Benchmarks (ToN_IoT, BoT-IoT, NSL-KDD, CICIDS2017)

#### 7. Booij et al. (IEEE Internet of Things Journal, 2022) — Dataset Heterogeneity & Standardization
- **Bibliographic Details**: T. M. Booij, I. Chiscop, E. Meeuwissen, N. Moustafa, and F. T. H. den Hartog, "ToN_IoT: The Role of Heterogeneity and the Need for Standardization of Features and Attack Types in IoT Network Intrusion Datasets," *IEEE Internet of Things Journal*, vol. 9, no. 1, pp. 485–496, 2022. DOI: `10.1109/JIOT.2021.3085194`.
- **Dataset Evaluated**: TON_IoT, BoT-IoT, UNSW-NB15, NSL-KDD.
- **Key Findings & Methodological Value**: 
  - Conducted rigorous statistical and ML audits demonstrating that earlier datasets (like BoT-IoT) yielded artificially inflated $>99.9\%$ accuracies due to lack of device heterogeneity and severe flood skew.
  - Demonstrated that models achieving $>99.6\%$ accuracy in-domain collapse when evaluated across datasets due to feature distribution shifts.
  - Identified **static identifier leakage**: network captures often contain hardcoded static IPs and MAC addresses for attacking/victim machines, causing models to memorize host identities rather than behavioral attack dynamics.
- **Limitations**: Relies on Zeek/NetFlow post-mortem flow records that require flow termination (or 60-second timeouts), which is too late to prevent line-rate volumetric compromises. Did not propose an operational filtering pipeline or evaluate CICIoT2023.

#### 8. Gulzar & Mustafa (Int. J. Machine Learning & Cybernetics, 2025) — DeepCLG Hybrid Model
- **Bibliographic Details**: Q. Gulzar and K. Mustafa, "Enhancing network security in industrial IoT environments: a DeepCLG hybrid learning model for cyberattack detection," *International Journal of Machine Learning and Cybernetics*, vol. 16, 2025. DOI: `10.1007/s13042-025-02544-w`.
- **Dataset Evaluated**: TON_IoT, BoT-IoT, NSL-KDD, UNSW-NB15.
- **Key Findings & Flaws**:
  - Proposed DeepCLG, stacking 1D-CNN, LSTM, and GRU layers in series.
  - While achieving 99.82% accuracy on the flood-dominated BoT-IoT benchmark, its detection rate **collapsed to 57.77% on the realistic UNSW-NB15 benchmark (a 42.05% drop)**.
  - Stacking multiple recurrent layers creates tens of milliseconds of inference latency, which is incompatible with deterministic sub-10 ms cycle times on industrial IoT controllers.

#### 9. Al-Hejri et al. (Scientific Reports, 2026) — Explainable Hybrid CNN-Transformer
- **Bibliographic Details**: A. M. Al-Hejri, R. M. Al-Tam, A. H. Sable, S. S. Alshamrani, K. M. Alshmrany, and A. Alshehri, "Improving IoT security through an explainable hybrid CNN-transformer model and federated learning," *Scientific Reports*, vol. 16, art. 24247, 2026. DOI: `10.1038/s41598-026-54935-w`.
- **Dataset Evaluated**: TON_IoT, BoT-IoT, IoT-23, NSL-KDD.
- **Key Findings & Flaws**:
  - Proposed combining 1D-CNN feature extraction with Transformer multi-head self-attention under federated learning.
  - Self-attention scales quadratically $O(N^2)$ in compute and memory, exhausting buffer memory on low-power IoT gateways.
  - While reporting 94.89% accuracy under balanced IID settings, accuracy dropped to **69.35% under realistic non-IID client distributions**, demonstrating that federated attention fails when edge devices experience disparate threat profiles.

#### 10. Essop et al. (Sensors, 2021) — IoT Dataset Generation & Evaluation Survey
- **Bibliographic Details**: I. Essop, J. C. Ribeiro, M. Papaioannou, G. Zachos, G. Mantas, and J. Rodriguez, "Generating Datasets for Anomaly-Based Intrusion Detection Systems in IoT and Industrial IoT Networks," *Sensors*, vol. 21, no. 4, art. 1528, 2021. DOI: `10.3390/s21041528`.
- **Key Findings**: Systematically surveyed IoT dataset generation methodologies. Pointed out that simulated environments (NS-2, NS-3, Cooja) lack physical hardware jitter and real background noise, and critiqued the continued reliance on obsolete 1999 KDD taxonomies (Probe, DoS, U2R, R2L). Highlighted the necessity of physical testbeds (which CICIoT2023 later implemented).

#### 11. Fenanir & Semchedine (Revue d'Intelligence Artificielle, 2023) — Smart Edge IDS via FL
- **Bibliographic Details**: S. Fenanir and F. Semchedine, "Smart Intrusion Detection in IoT Edge Computing Using Federated Learning," *Revue d'Intelligence Artificielle*, vol. 37, no. 5, pp. 1195–1207, 2023. DOI: `10.18280/ria.370505`.
- **Key Findings & Flaws**: Evaluated federated CNN/LSTM/GRU models, reporting 99.99% accuracy on the obsolete 1999 NSL-KDD benchmark, which plummets to 81.33% on UNSW-NB15. Ignored wireless uplink communication overhead across low-power mesh networks (Zigbee/LoRa).

#### 12. Moustafa (Sustainable Cities and Society, 2021) — The Network TON_IoT Datasets
- **Bibliographic Details**: N. Moustafa, "A new distributed architecture for evaluating AI-based security systems at the edge: Network TON_IoT datasets," *Sustainable Cities and Society*, vol. 72, art. 102994, 2021. DOI: `10.1016/j.scs.2021.102994`.
- **Key Findings & Flaws**: Introduced the TON_IoT testbed dataset. Evaluated Random Forest and LSTM baselines (~98.5% accuracy). As demonstrated by subsequent audits (Booij et al., 2022), the dataset suffered from static IP/MAC address leakage, allowing models to memorize attacker IP subnets rather than intrusion behavior.

#### 13. Qaddos et al. (Scientific Reports, 2024) — FW-SMOTE + Deep Attention Architecture
- **Bibliographic Details**: A. Qaddos, M. U. Yaseen, A. S. Al-Shamayleh, M. Imran, A. Akhunzada, and S. Z. Alharthi, "A novel intrusion detection framework for optimizing IoT security," *Scientific Reports*, vol. 14, art. 21789, 2024. DOI: `10.1038/s41598-024-72049-z`.
- **Key Findings & Flaws**: Proposed Feature-Weighted SMOTE (FW-SMOTE) combined with 1D-CNN + BiLSTM + Multi-Head Self-Attention, claiming 99.60% accuracy on BoT-IoT. High accuracy is a direct artifact of BoT-IoT's $>99.5\%$ flood imbalance; the multi-stage deep learning pipeline requires server-grade GPUs and cannot run at line rate on edge gateways.

#### 14. Umar et al. (Journal of Cloud Computing, 2025) — Energy-Efficient DNN-KDQ
- **Bibliographic Details**: F. Umar, M. Yasmeen, et al., "Energy-efficient deep learning-based intrusion detection system for edge computing: a novel DNN-KDQ model," *Journal of Cloud Computing*, vol. 14, art. 32, 2025. DOI: `10.1186/s13677-025-00762-9`.
- **Key Findings & Flaws**: Evaluated Knowledge Distillation and INT8 Quantization to compress a DNN from 196.77 KB to 51 KB, reporting 99.43% test accuracy on CICIDS2017. However, CICIDS2017 suffers from well-documented ground-truth flaws (inverted timestamps, corrupted flow calculations), INT8 quantization rounds off subtle decision boundaries on rare attacks, and the evaluation included zero modern IoT protocols (MQTT, CoAP).

---

## 4. Synthesis of Methodological Weaknesses in Existing IoT IDS Literature

Across all 14 surveyed studies, six systemic methodological weaknesses repeatedly undermine published claims:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│               SIX SYSTEMIC METHODOLOGICAL WEAKNESSES IN IoT IDS LITERATURE             │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. THE ACCURACY PARADOX                                                                │
│    Published headline accuracies exceeding 98.5%–99.9% are mathematically driven by    │
│    volumetric flood dominance (>85% DDoS/DoS). Minority exploit recall (brute-force,   │
│    web attacks, injection) frequently collapses to near-zero (<15%) levels.             │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 2. SYNTHETIC SMOTE CONTAMINATION                                                       │
│    Studies heavily employ SMOTE to artificially rebalance training data. Applying      │
│    k-NN linear interpolation to network flow features generates physically impossible  │
│    protocol dynamics (e.g., fractional packet counts, invalid TCP flag combinations).  │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 3. PREDISPOSITION TO FEATURE LEAKAGE VIA RANDOM SPLITTING                             │
│    Standard random 80/20 train/test splits disperse duplicate and identical-feature    │
│    vectors across both partitions. Without atomic feature-hash grouping, models can   │
│    memorize identical feature vectors rather than learning generalizable patterns.     │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 4. UNACKNOWLEDGED FEATURE-LABEL AMBIGUITY (LABEL COLLISIONS)                           │
│    Thousands of flows share identical feature vectors but carry contradictory labels   │
│    (e.g., Benign vs Flood). No surveyed study identifies or measures the impact of     │
│    these conflicting vectors on model performance.                                     │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 5. ABSENCE OF TEMPORAL CROSS-SESSION VALIDATION                                        │
│    Models are evaluated exclusively on in-distribution splits from the same capture   │
│    period. None of the surveyed studies evaluate models against distinct, subsequent  │
│    temporal capture sessions from the same physical testbed.                           │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 6. THE "EDGE DEEP LEARNING" LATENCY CONTRADICTION                                      │
│    Proposals for edge IoT security frequently deploy multi-layer recurrent stacks,     │
│    Transformers, or complex deep ensembles requiring tens of milliseconds of compute,  │
│    rendering them incapable of operating at line rate on resource-constrained devices. │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Comparative Evaluation: How Our 21-Model Study Compares with Existing Literature

Our investigation establishes an empirical benchmark on the CICIoT2023 dataset designed specifically to overcome the vulnerabilities documented above:

| Evaluation Dimension | Existing CICIoT2023 Literature | Our 21-Model Investigation |
|---|---|---|
| **Architectural Breadth** | Typically benchmarks 1–3 model families in isolation (e.g., trees only or deep learning only). | Evaluates **5 distinct model families** (SGD Linear, Random Forest, LightGBM, XGBoost, PyTorch MLP) under identical preprocessing and scaling. |
| **Diagnostic Granularities** | Often evaluates only binary or lumps flood classes in coarse multiclass groupings. | Evaluates **3 progressive granularities**: Binary (2 classes), Attack Category (8 classes), and Fine-Grained Attribution (34 classes). |
| **Class Imbalance Strategy** | Heavy reliance on synthetic oversampling (SMOTE) or unweighted empirical risk minimization. | **Zero Synthetic Injection**: Preserves 100% authentic physical packet dynamics; evaluates **cost-sensitive balanced weighting vs unweighted optimization**. |
| **Feature Duplication Audit** | Strips exact duplicate full rows at most; ignores redundant feature vectors. | Forensic audit identifies **28,464 exact duplicates (5.38%)** and **31,172 identical-label redundant feature vectors (5.89%)**. |
| **Conflicting Label Collisions** | Completely unacknowledged across all surveyed literature. | Discovered **8,740 unique conflicting feature hashes across 28,608 rows (5.40%)**; established Strategy 1 vs Strategy 2 (purged) ablation. |
| **Partitioning Integrity** | Standard random train/test splits or $k$-fold cross-validation. | **Atomic Feature-Hash Grouped Partitioning**: Guarantees zero feature overlap between train, val, and test splits ($\text{Overlap} = 0$). |
| **Evaluation Scope** | In-distribution random holdouts only. | **Dual-Partition Protocol**: Evaluated on both an **Internal In-Distribution Test Set ($N=75,105$)** and a **Same-Testbed Temporal Cross-Capture Test Set ($N=57,280$)**. |
| **Operational Feasibility** | Claims "edge efficiency" without wall-clock latency benchmarking at line rate. | Benchmarks single-flow latency distributions (P50/P95) and batch throughput, proving LightGBM ($220.4\ \mu\text{s}$ P50, $>930$k flows/sec) and SGD ($0.20\ \mu\text{s}$) edge viability. |

---

## 6. The Research Gap Addressed by This Project

### 6.1 Definition of the Primary Research Gap
While the literature on IoT intrusion detection has proliferated, prior work has operated under an unexamined assumption of **clean tabular separability** and has relied on **in-distribution random validation**. Specifically, prior research on the CICIoT2023 benchmark has not addressed:
1. **The Inter-Partition Feature Leakage Gap**: The presence of duplicate feature vectors across the dataset means that standard random splits can leak identical feature instances into test sets. Prior studies have not isolated feature vectors to evaluate whether models generalize when identical feature vectors are strictly prevented from bridging partitions.
2. **The Label Ambiguity Gap**: High-throughput network traffic generates identical statistical flow summaries for both benign background tasks and malicious packet floods. Prior studies treat conflicting ground-truth labels as ordinary stochastic noise, without quantifying how many feature hashes collide or measuring classifier behavior on ambiguous subsets.
3. **The Temporal Stability Gap**: Published models report high accuracy on in-distribution test splits, but their stability across distinct, subsequent recording sessions on the same testbed has remained unmeasured.
4. **The Minority Trade-Off Gap**: The quantitative trade-offs between aggregate accuracy, false-negative rates to benign traffic, and minority attack recall under cost-sensitive weighting versus unweighted optimization have not been systematically characterized across linear, tree, and neural architectures under leak-free conditions.

### 6.2 Distinctive Contributions vs. Standard Engineering Practices

To maintain scientific integrity and avoid unsupported novelty claims, our contributions are strictly separated into **distinctive research contributions** and **standard engineering practices**:

```
[DISTINCTIVE METHODOLOGICAL & EMPIRICAL CONTRIBUTIONS]
 ├── Forensic quantification of redundancy: Identified 28,464 exact duplicates (5.38%), 
 │   31,172 identical-label redundant vectors (5.89%), and 28,608 conflicting collision rows 
 │   across 8,740 unique hashes (5.40%) in the stratified CICIoT2023 sample.
 ├── Atomic feature-hash grouped partitioning protocol guaranteeing zero feature-vector overlap
 │   across train, validation, and test partitions (Overlap = 0).
 ├── Invariant common clean validation benchmark (N = 72,390) enabling direct, controlled 
 │   evaluation of collision purging without partition variance.
 ├── Empirical measurement of classifier behavior on ambiguous feature vectors: Models achieved 
 │   ~41.7% observed accuracy on the ambiguous validation subset (compared to 75.6% on clean data).
 ├── Empirical demonstration that collision purging (Strategy 2) yields a marginal numerical 
 │   cross-capture Macro-F1 improvement (+0.0013) while maintaining comparable in-distribution metrics.
 ├── Systematic quantification of class weighting trade-offs: Unweighted training increases 
 │   aggregate accuracy (72.65% → 74.45%) but increases false negatives to Benign by +125.3% 
 │   and cuts Tier 4 Web-tier recall from 34.63% to 16.81%.
 └── Same-testbed temporal cross-capture evaluation on 57,280 samples from Merged26–Merged40, 
     documenting cross-session stability across functional threat tiers.

[STANDARD ENGINEERING PRACTICES]
 ├── Preprocessor parameter serialization (models_saved_v2/preprocessor_strategy*.joblib) 
 │   ensuring zero-refit transformation during evaluation.
 ├── Standardization of continuous features via z-score scaling.
 ├── Single-pass, read-only test evaluation runner preventing parameter tuning on test sets.
 └── Logging of CPU wall-clock latency, memory consumption, and batch throughput.
```

---

## 7. Connection to Stage 6 Research Findings and Synthesis

This literature review and research gap analysis directly grounds and corroborates the empirical findings synthesized in `results_v2/STAGE_6_RESEARCH_SYNTHESIS.md`:

1. **Alignment on High Aggregate Accuracy**: Literature reporting $>98\%$ accuracy on CICIoT2023 (Neto et al. 2023, Khan et al. 2026, Almahaqeri et al. 2026) is explained by the flood dominance of the dataset. When our models are evaluated under class-weighted conditions, fine-grained accuracy reaches **$72.65\%$** (balanced LightGBM) and **$74.45\%$** (unweighted LightGBM), reflecting honest evaluation without synthetic SMOTE inflation.
2. **Alignment on Task-Specific Model Rankings**: Tree ensembles consistently lead across tabular network flow tasks: Random Forest leads Binary internal test Macro-F1 (**$0.7690$**), while LightGBM leads Category (**$0.6894$**) and Fine-Grained (**$0.6145$**). Deep learning (PyTorch MLP, 5 epochs) achieves lower Macro-F1 (**$0.5216$**), corroborating literature findings (Ullah et al., Booij et al.) that complex neural architectures provide no tabular performance advantage over gradient boosted trees while incurring significantly higher inference latencies.
3. **Alignment on Minority Attack Vulnerability**: Our Stage 5 error analysis confirmed that balanced LightGBM achieved an internal-test Web-tier (Tier 4) Macro-F1 of **$0.1748$**, and experienced false negatives primarily on OS reconnaissance scans ($21.06\%$ misclassified as Benign). This directly matches the empirical vulnerabilities first observed in Neto et al.'s Table 7 and Moussaoui et al.'s observations regarding the payload-blindness of statistical flow features.
4. **Resolution of Literature Overclaims**:
   - Rather than claiming that published results above 99% were caused by leakage, our literature review states that feature redundancy and unpartitioned random splits create conditions under which data leakage can inflate evaluation metrics.
   - Rather than claiming "zero overfitting," our study notes small validation-to-internal-test Macro-F1 differences ($|\Delta_{\text{gen}}| \le 0.0079$), which reflects consistent sampling across internal splits without proving the complete absence of overfitting.
   - Rather than describing temporal cross-capture testing as independent out-of-domain generalization, we consistently describe the Merged26–Merged40 holdout as **same-testbed temporal cross-capture evaluation**.

---

## 8. Conclusion

This comprehensive literature review and research gap analysis examined 14 peer-reviewed studies alongside internal technical documentation. The analysis establishes that while published studies routinely report near-perfect detection metrics on the CICIoT2023 and related IoT benchmarks, these results frequently rely on synthetic SMOTE oversampling, in-distribution random splits susceptible to feature leakage, and aggregate accuracy metrics that conceal near-zero recall on stealthy attack vectors.

Furthermore, none of the surveyed studies identified or controlled for identical feature vectors carrying conflicting ground-truth labels (5.40% of the stratified sample), and none evaluated temporal stability across distinct capture sessions from the same testbed.

By implementing atomic feature-hash grouped partitioning, an invariant clean validation benchmark, cost-sensitive balanced weighting without synthetic data injection, and dual-partition evaluation (internal test vs same-testbed temporal cross-capture), our 21-model study addresses these critical gaps and establishes a rigorous, reproducible empirical benchmark for IoT intrusion detection research.

---

## 9. References (Surveyed Research Archive)

1. **Khan, M. Z., Alam, M., Alam, I., Bokhari, M. U., Ashraf, Z., Khan, M. Z., & Masoodi, F. S.** (2026). A resource efficient IoT intrusion detection model using hybrid feature selection for edge computing. *Scientific Reports*, 16, 24187. https://doi.org/10.1038/s41598-026-59156-9.
2. **Almahaqeri, S. A., Almourish, M. H., Nasser, A. A., Alghawli, A. S. A., Elsayed, A. A. K., & Alhejoj, A. N.** (2026). An optimized gradient boosting framework for IoT intrusion detection: a comprehensive evaluation on the CICIoT2023 dataset. *Preprint / Manuscript Under Review*.
3. **Neto, E. C. P., Dadkhah, S., Ferreira, R., Zohourian, A., Lu, R., & Ghorbani, A. A.** (2023). CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environment. *Sensors*, 23(13), 5941. https://doi.org/10.3390/s23135941.
4. **Ullah, I., Wu, N.-C., Lin, C.-F., Kamal, M. S., & Mostafa, S. M.** (2026). Comparative analysis of deep learning and traditional methods for IoT botnet detection using a multi-model framework across diverse datasets. *Scientific Reports*, 16, 24190. https://doi.org/10.1038/s41598-026-59142-1.
5. **Gulzar, Q., & Mustafa, K.** (2025). Enhancing network security in industrial IoT environments: a DeepCLG hybrid learning model for cyberattack detection. *International Journal of Machine Learning and Cybernetics*, 16. https://doi.org/10.1007/s13042-025-02544-w.
6. **Abdelrahman, A., Elhossieny, A., Bassiony, I., & Mattar, A.** (2026). Feature-Efficient Machine Learning Framework for Binary and Multiclass Intrusion Detection in IoT Networks. *2026 ICEENG International Conference for Innovations in Intelligent Computing and Cybersecurity (IICC)*, IEEE. https://doi.org/10.1109/IICC69623.2026.11582694.
7. **Al-Hejri, A. M., Al-Tam, R. M., Sable, A. H., Alshamrani, S. S., Alshmrany, K. M., & Alshehri, A.** (2026). Improving IoT security through an explainable hybrid CNN-transformer model and federated learning. *Scientific Reports*, 16, 24247. https://doi.org/10.1038/s41598-026-54935-w.
8. **Booij, T. M., Chiscop, I., Meeuwissen, E., Moustafa, N., & den Hartog, F. T. H.** (2022). ToN_IoT: The Role of Heterogeneity and the Need for Standardization of Features and Attack Types in IoT Network Intrusion Datasets. *IEEE Internet of Things Journal*, 9(1), 485–496. https://doi.org/10.1109/JIOT.2021.3085194.
9. **Essop, I., Ribeiro, J. C., Papaioannou, M., Zachos, G., Mantas, G., & Rodriguez, J.** (2021). Generating Datasets for Anomaly-Based Intrusion Detection Systems in IoT and Industrial IoT Networks. *Sensors*, 21(4), 1528. https://doi.org/10.3390/s21041528.
10. **Fenanir, S., & Semchedine, F.** (2023). Smart Intrusion Detection in IoT Edge Computing Using Federated Learning. *Revue d'Intelligence Artificielle*, 37(5), 1195–1207. https://doi.org/10.18280/ria.370505.
11. **Moussaoui, J.-E., Kmiti, M., Maleh, Y., & El Gholami, K.** (2026). Resilient federated intrusion detection with explainable AI: a robust CNN-LSTM architecture for extreme non-IID data distributions. *Scientific Reports*, 16, 32367. https://doi.org/10.1038/s41598-026-61994-6.
12. **Moustafa, N.** (2021). A new distributed architecture for evaluating AI-based security systems at the edge: Network TON_IoT datasets. *Sustainable Cities and Society*, 72, 102994. https://doi.org/10.1016/j.scs.2021.102994.
13. **Qaddos, A., Yaseen, M. U., Al-Shamayleh, A. S., Imran, M., Akhunzada, A., & Alharthi, S. Z.** (2024). A novel intrusion detection framework for optimizing IoT security. *Scientific Reports*, 14, 21789. https://doi.org/10.1038/s41598-024-72049-z.
14. **Umar, F., Yasmeen, M., et al.** (2025). Energy-efficient deep learning-based intrusion detection system for edge computing: a novel DNN-KDQ model. *Journal of Cloud Computing*, 14, 32. https://doi.org/10.1186/s13677-025-00762-9.
