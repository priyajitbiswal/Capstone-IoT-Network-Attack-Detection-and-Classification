# STAGE 2: DATASET INTEGRITY & LEAKAGE INVESTIGATION REPORT
**Project Title**: IoT Network Attack Detection and Classification (CICIoT2023)  
**Auditor / Lead Researcher**: Senior ML & Cybersecurity Research Auditor  
**Date**: November 3, 2026  
**Status**: Stage 2 Complete — Awaiting User Approval Before Stage 3  

---

## Executive Summary

Pursuant to the approved Stage 1 Audit and the user's specific Stage 2 conditions, an exhaustive empirical investigation was executed on dataset integrity, duplicate topologies, cross-capture provenance, and leakage mechanisms. 

Key outcomes:
1. **Separation and Characterization of Duplicate Types**: Differentiated exact duplicate rows (28,464 instances, 5.38%), redundant feature vectors (38,140 instances, 7.20%), and conflicting label collisions (8,740 unique vectors, 28,608 rows, 5.40%). Identified their physical causes (automated packet flood repetition and protocol ambiguity caused by stripped IP headers).
2. **Deterministic, Leak-Free 70/15/15 Partitioning**: Implemented deterministic feature-hash group partitioning on deduplicated clean rows (501,098 flows). Formally verified **exactly 0.00% feature overlap between Train (350,821), Validation (75,172), and Test (75,105)**, while retaining all 34 classes across every partition.
3. **Cross-Capture Provenance & Coverage Verified**: Crawled files `Merged26.csv` through `Merged63.csv` (38 files, 26,376,916 flows). Confirmed that **all 34 classes are fully represented**. Extracted an independent, clean holdout evaluation dataset of **57,280 flows** (`cross_capture_test_clean.csv`) with **0.00% overlap with Train/Val/Test**.
4. **Corrected Preprocessing & XGBoost Weighting**: Engineered the isolated revised pipeline (`track1_dataset_evaluation/src/`, originally `src_v2/`). Corrected feature cleaning and `StandardScaler` to fit strictly on training data, and resolved the XGBoost class weighting flaw via `BalancedXGBClassifier`.

---

## 1. Deep-Dive Investigation of Duplicate Types

A granular census was performed on [`data/sample_stratified.csv`](file:///c:/Users/priya/Downloads/capstone/data/sample_stratified.csv) (529,562 flows $\times$ 39 features + 1 label). Three distinct categories of duplication exist:

```
Total Records in sample_stratified.csv: 529,562
│
├── Unique Feature Vectors: 491,422
│   ├── Unambiguous Vectors (1 Label Only): 482,682 (98.22%)
│   └── Ambiguous Vectors (>1 Conflicting Label): 8,740 (1.78%)
│       └── Rows involved in cross-class conflicts: 28,608 (5.40%)
│
└── Redundant Duplicate Instances: 38,140 (7.20%)
    ├── Exact Duplicate Rows (Identical Features AND Label): 28,464 (5.38%)
    └── Cross-Class Duplicate Instances (Same Features, Different Label): 9,676 (1.83%)
```

### A. Exact Duplicate Rows (Features AND Label Identical)
- **Count**: 28,464 redundant rows (47,430 rows belong to exact duplicate clusters).
- **Concentration**: Overwhelmingly concentrated in volumetric flooding attacks:
  - `DDOS-ICMP_FLOOD`: 5,630 rows in duplicate clusters (**28.1% of class**)
  - `DDOS-TCP_FLOOD`: 5,619 rows (**28.1% of class**)
  - `DDOS-RSTFINFLOOD`: 4,995 rows (**25.0% of class**)
  - `DDOS-SYNONYMOUSIP_FLOOD`: 4,755 rows (**23.8% of class**)
  - `DOS-TCP_FLOOD`: 4,568 rows (**22.8% of class**)
  - `DDOS-UDP_FLOOD`: 4,564 rows (**22.8% of class**)
- **Minority Class Behavior**: Rare, complex attacks exhibit virtually zero exact duplicates:
  - `BENIGN`: Only 63 duplicate rows (**0.16% of class**)
  - `BACKDOOR_MALWARE`: Only 2 duplicate rows (**0.19% of class**)
  - `UPLOADING_ATTACK`: **0 duplicate rows (0.0%)**
  - `COMMANDINJECTION`: **0 duplicate rows (0.0%)**
  - `SQLINJECTION`: **0 duplicate rows (0.0%)**
- **Causes**: Attack generation tools (e.g., `hping3`, `scapy`, `LOIC`) flood targets with identical packet templates (fixed TCP window size, identical header length, constant TTL, zero payload). When sliding-window flow statistics are computed with IP addresses and timestamps stripped, consecutive flows yield numerically identical 39-dimensional feature vectors.
- **Consequences**: Under random 80/20 train/test splitting, identical flood flows fall into both Train and Test. The model memorizes exact feature coordinates, producing artificially inflated test accuracy on flooded classes while providing no insight into generalization.

### B. Duplicate Feature Vectors with Conflicting Labels
- **Count**: 8,740 unique feature vectors that carry multiple conflicting labels, spanning **28,608 rows (5.40% of the dataset)**.
- **Pairwise Conflict Breakdown**:
  1. **DoS vs. DDoS of Identical Protocol (Over 85% of all conflicts)**:
     - `DDOS-TCP_FLOOD` $\leftrightarrow$ `DOS-TCP_FLOOD`: **2,722 shared feature vectors**
     - `DDOS-SYNONYMOUSIP_FLOOD` $\leftrightarrow$ `DDOS-SYN_FLOOD`: **2,346 shared feature vectors**
     - `DDOS-SYN_FLOOD` $\leftrightarrow$ `DOS-SYN_FLOOD`: **1,936 shared feature vectors**
     - `DDOS-SYNONYMOUSIP_FLOOD` $\leftrightarrow$ `DOS-SYN_FLOOD`: **1,770 shared feature vectors**
     - `DDOS-UDP_FLOOD` $\leftrightarrow$ `DOS-UDP_FLOOD`: **1,333 shared feature vectors**
     - *Cause*: A single TCP SYN or UDP flood packet emitted by a single attacker (DoS) is physically identical to a packet emitted by one node of a botnet (DDoS). The defining difference between DoS and DDoS is distributed IP entropy. Because the dataset creators stripped IP addresses and timestamps, individual flow records of DoS and DDoS are mathematically identical.
  2. **Reconnaissance Sub-Vector Collisions**:
     - `RECON-OSSCAN` $\leftrightarrow$ `RECON-PORTSCAN`: **150 shared feature vectors**
     - *Cause*: Both port scanning and OS detection probes use identical raw TCP SYN/FIN probe sequences.
  3. **Benign vs. Attack Collisions (Minimal: 0.11% of dataset)**:
     - Only **41 unique feature vectors** (621 rows) exhibit conflicts between `Benign` and `Attack`.
     - 424 of those 621 rows are `MITM-ARPSPOOFING` $\leftrightarrow$ `BENIGN` (standard ARP traffic vs spoofed ARP reply).
- **Consequences**: 
  - Represents irreducible Bayes error / label noise at the 34-class level. A deterministic model given identical input $X$ cannot output two distinct classes.
  - If identical vectors appear in Train as class A and in Test as class B, the model is guaranteed to incur an error, distorting evaluation metrics.
  - In training, contradictory labels generate conflicting gradient updates, destabilizing probability calibration.

---

## 2. Comparison of Duplicate Handling Strategies & Recommendation

| Strategy | Description | Impact on Data Size | Leakage Elimination | Minority Impact | Scientific Defensibility |
|---|---|:---:|:---:|:---:|:---:|
| **Strategy A: Blind Deletion of All Duplicate Feature Vectors** | Delete all 38,140 duplicate vectors, retaining only strictly unique feature coordinates across the entire dataset. | Drops 38,140 rows (-7.2%). | High. | Discards legitimate high-frequency network behavior. | Moderate. Artificially purges ambiguous traffic that naturally exists. |
| **Strategy B: Exact Row Deduplication Only (Random Split)** | Delete 28,464 exact duplicates within each class, but split randomly. | Retains 501,098 rows (-5.4%). | **Fails**. Cross-class identical vectors still split across train and test, causing leakage. | Minimal. | Low. Does not eliminate feature-level leakage. |
| **Strategy C: Exact Deduplication + Hash-Based Group Partitioning (RECOMMENDED)** | 1. Remove 28,464 exact duplicate rows (preserving 1 unique copy per flow).<br/>2. Group identical feature vectors by 64-bit hash.<br/>3. Assign all rows of any given feature hash strictly to ONE partition (Train, Val, or Test). | Retains 501,098 rows (-5.38%). Preserves all physical flows. | **Complete (0.00% overlap)**. Mathematically guarantees zero feature leakage across partitions. | **Zero minority degradation**. Preserves 100% of unique minority attack signatures. | **Highest**. Standard in spatial/grouped ML; preserves ambiguous flows within partitions without leaking across them. |

### Evidence-Based Recommendation: Strategy C
We adopt **Strategy C**. By deduplicating exact redundant rows (reducing the sample from 529,562 to 501,098) and partitioning by feature vector hash:
1. Exact memorization of identical flows is completely eliminated.
2. If an ambiguous vector exists (e.g. DoS vs DDoS), all instances stay together in the *same* partition. If assigned to Train, it never appears in Test. If in Test, it was never seen in Train.
3. Feature overlap between Train, Validation, and Test is **mathematically zero**.

---

## 3. Verified Reproducible Partitioning Results

The partitioning algorithm was implemented in [`src_v2/data_partitioning.py`](file:///c:/Users/priya/Downloads/capstone/src_v2/data_partitioning.py).  
Original dataset file [`data/sample_stratified.csv`](file:///c:/Users/priya/Downloads/capstone/data/sample_stratified.csv) remains **100% untouched**.  
Assignments are exported to [`data/partitions/partition_assignments.csv`](file:///c:/Users/priya/Downloads/capstone/data/partitions/partition_assignments.csv) with provenance in [`data/partitions/partition_metadata.json`](file:///c:/Users/priya/Downloads/capstone/data/partitions/partition_metadata.json).

### Partition Summary Table

| Partition | Row Count | Percentage | Unique Feature Hashes | Classes Present | Overlap with Train | Overlap with Val | Overlap with Test |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Train** | **350,821** | **70.01%** | 343,995 | **34 of 34** | — | **0 (0.00%)** | **0 (0.00%)** |
| **Validation** | **75,172** | **15.00%** | 73,713 | **34 of 34** | **0 (0.00%)** | — | **0 (0.00%)** |
| **Test** | **75,105** | **14.99%** | 73,714 | **34 of 34** | **0 (0.00%)** | **0 (0.00%)** | — |
| **Total** | **501,098** | **100.00%** | **491,422** | **34 of 34** | **0** | **0** | **0** |

### Minority Class Retention Across Partitions

| Minority Attack Profile | Total Instances in Sample | Train Set (Target: 70%) | Validation Set (Target: 15%) | Test Set (Target: 15%) |
|---|:---:|:---:|:---:|:---:|
| `UPLOADING_ATTACK` | 416 | 312 (75.0%) | 44 (10.6%) | 60 (14.4%) |
| `RECON-PINGSWEEP` | 722 | 486 (67.3%) | 111 (15.4%) | 125 (17.3%) |
| `BACKDOOR_MALWARE` | 1,060 | 766 (72.3%) | 155 (14.6%) | 139 (13.1%) |
| `XSS` | 1,239 | 887 (71.6%) | 169 (13.6%) | 183 (14.8%) |
| `COMMANDINJECTION` | 1,666 | 1,186 (71.2%) | 225 (13.5%) | 255 (15.3%) |
| `SQLINJECTION` | 1,718 | 1,212 (70.5%) | 252 (14.7%) | 254 (14.8%) |
| `BROWSERHIJACKING` | 1,904 | 1,326 (69.6%) | 288 (15.1%) | 290 (15.2%) |
| `DICTIONARYBRUTEFORCE` | 4,181 | 2,933 (70.2%) | 638 (15.3%) | 610 (14.6%) |
| `BENIGN` | 39,944 | 28,016 (70.1%) | 5,969 (14.9%) | 5,959 (14.9%) |

---

## 4. Cross-Capture Investigation: `Merged26.csv` – `Merged63.csv`

### A. Provenance and Physical Storage
- **File Scope**: 38 pristine CSV files (`Merged26.csv` through `Merged63.csv`) totaling ~5.2 GB in [`MERGED_CSV/`](file:///c:/Users/priya/Downloads/capstone/MERGED_CSV/).
- **Total Flow Count**: **26,376,916 raw flows**.
- **Provenance**: In the CICIoT2023 dataset collection, 169 PCAP captures were recorded over several days and converted to CSVs. The repository's initial sampling script (`create_sample.py`) processed only `Merged01.csv` through `Merged25.csv`. Files 26 through 63 represent completely untouched network capture sessions.

### B. Class Coverage Verification
A full scan of all 26.37 million flows confirmed that **all 34 classes are present in `Merged26`–`Merged63`**, including substantial representation of all rare attack profiles:
- `BENIGN`: 615,853 flows
- `DICTIONARYBRUTEFORCE`: 7,335 flows
- `BROWSERHIJACKING`: 3,299 flows
- `COMMANDINJECTION`: 3,080 flows
- `SQLINJECTION`: 2,898 flows
- `XSS`: 2,171 flows
- `BACKDOOR_MALWARE`: 1,759 flows
- `RECON-PINGSWEEP`: 1,258 flows
- `UPLOADING_ATTACK`: 686 flows

### C. Construction of Independent Cross-Capture Test Set
Using [`src_v2/cross_capture.py`](file:///c:/Users/priya/Downloads/capstone/src_v2/cross_capture.py), we extracted a representative sample across `Merged26.csv`–`Merged40.csv`.
- **Initial extraction**: 63,423 flows across 34 classes.
- **Overlap check against Train**: Found 3,925 feature hashes (6.26%) that matched training feature vectors (caused by identical automated packet flooding scripts across recording days).
- **Leak-Free Filtering**: Purged overlapping hashes against Train/Val/Test (`cross_capture_metadata.json`) to create `cross_capture_test_clean.csv`:
  - **Clean flow count**: **57,280 flows** (`57,148` unique feature vectors; exact overlaps with Train = `0`, Val = `0`, Test = `0`).
  - **Class coverage**: **34 of 34 classes verified present**.

---

## 5. Preprocessing & Model Training Corrections

### A. Preprocessing Pipeline (`src_v2/preprocessing.py`)
1. **Zero-Leakage Cleaning**:
   - `LeakFreePreprocessor.fit()` scans `X_train` only to compute finite maxima for each column (e.g., `Rate` max = $3,355,443.2$).
   - `clean_transform()` applies these learned thresholds to validation and test partitions.
2. **Zero-Leakage Standardization**:
   - `StandardScaler` is fitted strictly on `X_train` (`fit_transform`).
   - `X_val` and `X_test` are transformed using the training mean and standard deviation.

### B. XGBoost Class Weighting Correction (`src_v2/models.py`)
- **Problem**: In `src/models.py`, `class_weight="balanced"` was passed to `xgb.XGBClassifier`, which is silently ignored by XGBoost's scikit-learn API without passing `sample_weight` into `.fit()`.
- **Correction**: Implemented `BalancedXGBClassifier` subclassing `xgb.XGBClassifier`. In `fit()`, if `class_weight == "balanced"` and `sample_weight is None`, it automatically calculates balanced sample weights via `compute_sample_weight("balanced", y)`:
  $$w_i = \frac{N}{K \cdot N_{y_i}}$$
  and passes them to the C++ engine. Cleanly filters constructor parameters to prevent runtime warnings.

---

## 6. Revised Pipeline Isolation

The revised Track 1 pipeline (`track1_dataset_evaluation/src/`, originally `src_v2/`) is cleanly separated from Track 2's baseline pipeline (`track2_ids_engineering/src/`), with hash partition maps and metadata saved under `track1_dataset_evaluation/data/partitions/` (`partition_assignments.csv`, `partition_metadata.json`, `cross_capture_test_clean.csv`, `cross_capture_metadata.json`, `ablation_strategy2/`).

---

## 7. Stage 2 Verification Checklist

- Raw `Merged01–63.csv` and `data/processed/sample_stratified.csv` kept strictly read-only; Strategy C (`70.01% / 15.00% / 14.99%`) verified at **0.00% feature overlap** across all 6 pairs; `LeakFreePreprocessor` and `BalancedXGBClassifier` unit-verified prior to Stage 3 execution.
