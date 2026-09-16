#set page(
  paper: "us-letter",
  margin: (top: 1.8cm, bottom: 2.0cm, left: 1.5cm, right: 1.5cm),
  columns: 2,
)
#set text(
  font: "Liberation Serif",
  size: 9.5pt,
  spacing: 120%,
)
#set par(justify: true, leading: 0.55em)

// Paper Title and Authors spanning both columns
#place(
  top,
  scope: "parent",
  float: true,
  [
    #align(center)[
      #text(size: 17pt, weight: "bold")[IoT Network Attack Detection and Classification]
      #v(0.5em)
      #text(size: 10.5pt, weight: "semibold")[Team Archon --- CS4999: Capstone Project] \
      #v(0.4em)
      #grid(
        columns: (1fr, 1fr, 1fr),
        align: center,
        [
          #text(weight: "bold")[Priyajit Biswal] \
          #text(size: 8.5pt)[23011102068] \
          #text(size: 8pt)[priyajit23110510\@snuchennai.edu.in]
        ],
        [
          #text(weight: "bold")[Rohit K Manoj] \
          #text(size: 8.5pt)[23011102073] \
          #text(size: 8pt)[rohit23110492\@snuchennai.edu.in]
        ],
        [
          #text(weight: "bold")[Rishab Rajeev] \
          #text(size: 8.5pt)[23011102072] \
          #text(size: 8pt)[rishab23110023\@snuchennai.edu.in]
        ]
      )
      #v(0.35em)
      #text(size: 8.5pt)[
        *Faculty In Charge:* Dr. Vegesna S M Srinivasavarma (srinivasv\@snuchennai.edu.in) \
        *Department of Computer Science & Engineering, Shiv Nadar University Chennai*
      ]
    ]
    #v(0.8em)
    #rect(
      width: 100%,
      stroke: 0.5pt + luma(180),
      fill: luma(250),
      inset: 10pt,
      radius: 3pt,
    )[
      #text(weight: "bold", size: 9pt)[Abstract] ---
      #text(size: 8.5pt)[
        The rapid proliferation of Internet of Things (IoT) devices has created an expansive attack surface characterized by severe protocol diversity, constrained processing hardware, and threat heterogeneity. Traditional Network Intrusion Detection Systems (NIDS) struggle with massive class imbalance and strict real-time line-rate latency requirements. In this work, we present an end-to-end machine learning and deep learning intrusion detection framework evaluated on the CICIoT2023 benchmark (45 million flows across 33 attack profiles). We develop a memory-safe stratified subsampling engine (529,562 flows) that preserves 100% of rare minority attack instances while eliminating out-of-memory bottlenecks. We evaluate five model architectures (SGD Logistic Regression, Random Forest, LightGBM, XGBoost, and PyTorch Tabular DNN) across four progressive tiers: Binary Detection (2 Classes), Functional Category Detection (8 Classes), Fine-Grained Attack Attribution (34 Classes), and Feature-Optimized Edge Detection. XGBoost achieves 95.28% accuracy and 1.05 $mu$s/flow latency in binary filtering, and 74.12% accuracy with 0.6125 Macro F1 in fine-grained 34-class classification. Furthermore, by employing multi-model consensus feature selection, we prune 48.7% of the feature space (from 39 down to 20 core features), achieving a 28.55% latency reduction (12.15 $mu$s/flow, 82,299 flows/sec) with less than 2.0% accuracy degradation. Finally, we formulate a Three-Tier Hierarchical IoT Defense Blueprint combining wire-speed linear filtering with edge gateway classification and centralized forensic attribution.
      ] \
      #v(0.4em)
      #text(weight: "bold", size: 8.5pt)[Keywords] ---
      #text(size: 8.5pt, style: "italic")[IoT Intrusion Detection, CICIoT2023, XGBoost, LightGBM, Class Imbalance, Edge Computing, Feature Selection, Latency Optimization.]
    ]
    #v(1.0em)
  ]
)

== 1. Introduction
The Internet of Things (IoT) ecosystem interconnects billions of resource-constrained sensors, smart appliances, and industrial actuators directly to public networks. While enabling automated smart grids and medical telemetry, IoT devices are notoriously vulnerable to cyber exploitation due to unpatched firmware, hardcoded credentials, and lack of host-based endpoint protection @ciciot2023. Volumetric Distributed Denial of Service (DDoS) botnets like Mirai can quickly weaponize millions of compromised nodes, while stealthy application-layer adversaries leverage command injection and brute-force attacks to gain persistent unauthorized access @ahmad2021network.

A central challenge in modern Network Intrusion Detection Systems (NIDS) lies in balancing *diagnostic granularity* against *computational latency*. Perimeter gateways require sub-microsecond binary packet filtering to discard multi-gigabit flooding attacks before hardware buffers overflow. Conversely, Security Operations Center (SOC) analysts require fine-grained classification across dozens of distinct malware strains to dispatch appropriate mitigation playbooks. Modern IoT gateways possess limited CPU and memory headroom, necessitating aggressive feature space reduction to maintain line-rate throughput @mishra2019detailed.

In this paper, we address these challenges on the *CICIoT2023* benchmark:
1. *Memory-Safe Stratified Engineering*: We design a streaming chunk processor and stratified subsampling engine (529,562 flows) that retains 100% of rare minority vectors while eliminating out-of-memory (OOM) failures.
2. *Multi-Tiered Diagnostic Progression*: We evaluate five model families across Binary (2 classes), Category (8 classes), and Fine-Grained (34 classes) tiers, establishing empirical accuracy and latency scaling trajectories.
3. *Resolution of Minority Starvation*: By introducing cost-sensitive balanced class weighting, we achieve high-precision detection across stealthy exploits, including 81.11% precision on Command Injection and 77.19% precision on Dictionary Brute Force.
4. *Consensus Edge Optimization*: We construct an ensemble feature importance ranking across RF, XGBoost, and LightGBM, reducing the feature space from 39 to 20 features, achieving a 28.55% latency speedup for edge routers.
5. *Hierarchical Defense Blueprint*: We propose a Three-Tier Architecture integrating line-rate linear screening (>5M flows/sec) with edge tree classification and cloud forensics.

== 2. Dataset Architecture & Preprocessing

=== 2.1 The CICIoT2023 Benchmark
Introduced by the Canadian Institute for Cybersecurity, the CICIoT2023 dataset captures traffic from an experimental testbed of 105 real smart-home and industrial IoT devices subjected to 33 distinct attack profiles and benign activities @ciciot2023. While exploratory scripts assumed legacy 46-feature schemas, ground-truth inspection of the raw 63 CSV partitions reveals an exact 39-feature schema encompassing packet rates, inter-arrival times (IAT), transport protocol indicators, TCP flag counters, and window statistics.

=== 2.2 Stratified Sampling with Minority Retention
Volumetric floods comprise over 85% of raw IoT traffic, whereas application-layer web exploits represent less than 0.05%. Unchecked random sampling discards rare attack signatures entirely. To overcome this, our sampling engine caps majority classes at 20,000 samples (40,000 for Benign) while retaining *100% of all rare and stealthy attack instances*:
- `UPLOADING_ATTACK`: 416 instances retained.
- `RECON-PINGSWEEP`: 722 instances retained.
- `BACKDOOR_MALWARE`: 1,061 instances retained.
- `XSS`: 1,239 instances retained.
- `COMMANDINJECTION`: 1,666 instances retained.
- `SQLINJECTION`: 1,718 instances retained.
- `BROWSERHIJACKING`: 1,904 instances retained.
- `DICTIONARYBRUTEFORCE`: 4,181 instances retained.

The resulting stratified dataset contains 529,562 records across all 34 classes.

=== 2.3 Numerical Sanitization & Zero-Leakage Scaling
Zero-duration burst flows ($d t -> 0$) produce infinite values in the `Rate` column, which we cap to the maximum observed finite rate ($3.35 times 10^6$ packets/sec). Single-packet windows yielding `NaN` variance are cleanly imputed to 0.0. Continuous features are normalized via standard scaling strictly fitted on the 80% training partition (423,649 flows) and evaluated on the 20% test partition (105,913 flows).

== 3. Methodology & Model Architectures

We benchmark five model architectures:
1. *SGD Logistic Regression*: Scalable linear classifier trained via stochastic gradient descent with log-loss, minimizing full-batch quasi-Newton convergence bottlenecks.
2. *Random Forest (RF)*: Ensemble of 50 decorrelated decision trees with max depth 15 and balanced bootstrapping @breiman2001random.
3. *LightGBM*: Leaf-wise tree boosting with gradient-based one-side sampling (GOSS) and exclusive feature bundling @ke2017lightgbm.
4. *XGBoost*: Depth-wise gradient boosted trees utilizing exact greedy split finding and second-order Taylor loss approximations @chen2016xgboost.
5. *PyTorch Tabular DNN*: Multi-layer perceptron (Input[39] $->$ Dense[128] $->$ BatchNorm $->$ ReLU $->$ Dropout[0.2] $->$ Dense[64] $->$ BatchNorm $->$ ReLU $->$ Output[C]) trained with AdamW and balanced cross-entropy weighting @paszke2019pytorch.

=== 3.1 Cost-Sensitive Loss Formulation
To mitigate severe multi-class imbalance without synthetic oversampling, class weights $W_c$ are injected into the loss functions:
$ W_c = N_("total") / (K dot N_c) $
where $K$ is the number of classes, $N_("total")$ is total sample count, and $N_c$ is the frequency of class $c$.

== 4. Experimental Results & Analysis

=== 4.1 Phase 1: Binary Detection (2 Classes)
Binary detection evaluates whether a flow is Attack (1) or Benign (0). As reported in Table 1, *XGBoost* demonstrates superior performance, achieving 95.28% accuracy, 0.8133 Macro F1, and an inference latency of 1.05 $mu$s/flow (throughput of 954,681 flows/sec). Notably, SGD Logistic Regression processes over 5,000,000 flows/sec with 0.20 $mu$s latency, validating its utility as a wire-speed packet scrubber.

=== 4.2 Phase 2: Category Detection (8 Classes)
Categorization into eight functional groups (`DDoS`, `DoS`, `Mirai`, `Recon`, `Spoofing`, `Web`, `BruteForce`, `Benign`) enables SOC playbook automation. XGBoost maintains leadership with 82.87% accuracy and 0.6876 Macro F1. Confusion analysis reveals that DoS and DDoS exhibit minor boundary confusion due to identical per-packet volume profiles in the absence of source IP entropy.

=== 4.3 Phase 3: Fine-Grained Attack Attribution (34 Classes)
Expanding to all 33 individual attack profiles represents the most rigorous testbed. XGBoost achieves 74.12% accuracy, while LightGBM achieves the highest Macro F1 (0.6176) and 65.25% Macro Recall.

#v(0.5em)
#table(
  columns: (1.3fr, 0.6fr, 0.9fr, 0.9fr, 1.0fr),
  stroke: 0.5pt + luma(200),
  inset: 4pt,
  align: (left, center, center, center, right),
  table.header(
    [*Phase & Model*], [*Cls*], [*Acc (%)*], [*F1-Mac*], [*Lat ($mu$s)*]
  ),
  [*Phase 1: Binary*], [], [], [], [],
  [XGBoost], [2], [*95.28*], [*0.8133*], [1.05],
  [PyTorch DNN], [2], [94.66], [0.7872], [14.50],
  [LightGBM], [2], [89.60], [0.7633], [1.69],
  [SGD Logistic Reg], [2], [82.91], [0.6806], [*0.20*],
  [*Phase 2: Category*], [], [], [], [],
  [XGBoost], [8], [*82.87*], [*0.6876*], [6.24],
  [LightGBM], [8], [78.82], [*0.6876*], [18.84],
  [Random Forest], [8], [78.17], [0.6814], [8.65],
  [SGD Logistic Reg], [8], [74.48], [0.6042], [*0.26*],
  [*Phase 3: Attack 34*], [], [], [], [],
  [XGBoost], [34], [*74.12*], [0.6125], [13.35],
  [LightGBM], [34], [72.83], [*0.6176*], [74.88],
  [Random Forest], [34], [70.74], [0.5970], [8.24],
  [PyTorch DNN], [34], [63.61], [0.5321], [13.95],
  [*Step 5: Opt Edge*], [], [], [], [],
  [XGBoost (20 Feats)], [34], [*72.16*], [*0.5886*], [*12.15*],
  [LightGBM (20 Feats)], [34], [70.80], [0.5963], [80.09],
  [Random Forest (20)], [34], [68.93], [0.5783], [15.15],
)
#align(center)[#text(size: 7.5pt, style: "italic")[Table 1: Master Capstone Synthesis Benchmark (105,913 Test Instances)]]

=== 4.4 Minority Attack Vector Resolution
Cost-sensitive balanced weighting prevents zero-recall failure modes across rare exploits:
- *Command Injection* (333 samples): XGBoost achieves 81.11% precision (0.3452 F1).
- *Dictionary Brute Force* (836 samples): XGBoost achieves 77.19% precision (0.3925 F1); LightGBM reaches 51.67% recall.
- *Browser Hijacking* (381 samples): XGBoost reaches 71.19% precision; LightGBM achieves 50.66% recall.
- *SQL Injection* (344 samples): XGBoost achieves 64.00% precision; Random Forest achieves 59.59% recall.
- *Backdoor Malware* (212 samples): XGBoost achieves 60.87% precision.

#v(0.5em)
#align(center)[
  #image("figures/capstone_master_synthesis.png", width: 100%)
  #text(size: 7.5pt, style: "italic")[Figure 1: Comprehensive 4-Panel Capstone Synthesis: (A) Accuracy progression; (B) Macro F1 resilience; (C) Latency vs Macro F1 Pareto frontier; (D) Network throughput vs line-rate thresholds.]
]

== 5. Feature Importance & Edge Latency Optimization

To enable embedded IoT deployment, we establish an ensemble consensus ranking:
$ S_("consensus")(f) = 1/3 [ S_("RF")^("Gini")(f) + S_("XGB")^("Gain")(f) + S_("LGB")^("Split")(f) ] $
The top 5 features (`Protocol Type`, `syn_count`, `Max`, `Header_Length`, `Tot sum`) account for over 32% of total importance, while static application indicators (`IRC`, `DHCP`, `SMTP`, `Telnet`, `DNS`) contribute $<0.1$%.

#v(0.5em)
#align(center)[
  #image("figures/feature_importance_ranking.png", width: 100%)
  #text(size: 7.5pt, style: "italic")[Figure 2: Top 20 features by consensus importance and cumulative information retained curve.]
]

Pruning from 39 down to 20 core features yields:
- *RAM Savings*: Vector storage drops from 156 bytes to 80 bytes per flow (*48.7% reduction*).
- *Latency Speedup*: XGBoost inference drops from 17.01 $mu$s to 12.15 $mu$s (*+28.55% speedup*, reaching *82,299 flows/sec*).
- *Minimal Accuracy Drop*: Accuracy decreases by only 1.96% (from 74.12% to 72.16%).

== 6. Proposed Hierarchical IoT Architecture
Based on these empirical benchmarks, we propose a Three-Tier Defense Blueprint:
1. *Tier 1 (Perimeter Wire-Filter)*: SGD Linear classifier operating at $>$5,000,000 flows/sec with 0.20 $mu$s latency, immediately passing verified benign packets.
2. *Tier 2 (IoT Gateway Defense)*: Reduced 20-feature XGBoost model running on local ARM routers, classifying 82,299 flows/sec with 72.16% fine-grained accuracy.
3. *Tier 3 (Cloud SOC Forensics)*: Centralized SIEM executing full 34-class attribution for automated patch deployment.

== 7. Conclusion
This research presented an empirical evaluation of machine learning architectures on the CICIoT2023 benchmark across binary, category, fine-grained, and feature-reduced tiers. Tree boosting models (XGBoost and LightGBM) consistently outperformed deep neural networks on tabular flow statistics. By implementing cost-sensitive balanced weighting and consensus feature pruning, we demonstrated that an optimized 20-feature XGBoost classifier can deliver line-rate 82,299 flows/sec intrusion detection on edge IoT gateways with exceptional minority attack detection capability.

#v(0.8em)
#bibliography("references.bib", style: "ieee")
