# Stage 8 Complete Report: Research Consolidation, Certified Reconciliation & Manuscript Packaging

> Consolidates Stages 8A, 8A.1, 8A.2, 8B, 8C, and 8D into one certified Track 1 reference. Every numerical constant below is reconciled directly against `data/processed/sample_stratified.csv` and `results/tables/{centralized_experiment_ledger,stage4_test_results}.csv` with all legacy audit discrepancies permanently resolved.

---

## 1. Authoritative Paper Framing, Central RQ & Four Contributions

- **Paper Title**: *Unmasking Data Leakage and Temporal Fragility in IoT Network Intrusion Detection: An Empirical Audit and Benchmark on CICIoT2023* *(published title in `paper.typ` / `main.tex`: "Empirical Evaluation of Data Leakage, Class Imbalance, and Temporal Fragility in IoT Network Intrusion Detection on CICIoT2023")*
- **Central Research Question**: *When identical-feature data leakage and class imbalance artifacts are rigorously controlled through atomic hash-grouped partitioning and cost-sensitive optimization, what is the realistic baseline diagnostic capability, minority attack recall, and temporal stability of standard machine learning architectures on CICIoT2023?*
- **Four Consolidated Contributions**:
  1. **Forensic Redundancy, Collision Taxonomy & Random-Split Leakage Audit** across `529,562` flows.
  2. **Reproducible Atomic SHA-256 Group-Hash Partitioning Standard** (`Train ∩ Val = Train ∩ Test = Val ∩ Test = ∅`) with strictly fit-on-train preprocessors.
  3. **Dissection of Collision Ambiguity vs. Model Boundary Stability** via an invariant clean validation slice (`N = 72,390`, `Δ = +0.0022`).
  4. **Quantified Security Trade-Offs**: Unweighted minority recall collapse (`-79.1%` to `-83.7%` SQLi, `+125.3%` Benign FNs) vs. same-testbed temporal fragility across `Merged26–40`.

---

## 2. Certified Duplication Arithmetic & Random-Split Leakage Proofs (Stage 8A.2)

### 2.1 Reconciliation of `28,608` (Pre-Dedup) vs. `18,416` (Post-Dedup) & Benign Collision Taxonomy
| Metric Dimension | Full Sample (`529,562`) | Unique Stratum (`501,098`) | Exact Reconciliation |
|---|---:|---:|---|
| **Exact Duplicate Rows ($\mathcal{E}$)** | **28,464** (`5.375%`) | **0** (`0.00%`) | Dropped in creating unique stratum |
| **Distinct 39-Feature Coordinates ($\mathcal{U}$)** | **491,422** | **491,422** | Invariant (`469,782` singletons + `21,640` repeated in full) |
| **Consistent-Label Repeated Hashes ($\mathcal{U}_{\text{cons}}$)** | `12,900` (`31,172` rows) | `0` hashes (`12,900` rows) | Each collapses to 1 singleton row (`-18,272` copies) |
| **Conflicting-Label Collision Hashes ($\mathcal{U}_{\text{conf}}$)** | **8,740** (**28,608** rows, `5.402%`) | **8,740** (**18,416** rows, `3.675%`) | Contains `10,192` within-class dups (`28,608 - 10,192 = 18,416`) |
| **Total Repeated-Vector Rows** | **59,780** (`11.29%`) | **18,416** (`3.68%`) | *Retracts legacy Stage 1 "78.9% / 417,824" miscount* |
| **Strategy 2 Collision-Purged Set** | — | **482,682** | `501,098 - 18,416` (`Train=337,838`, `Val=72,390`, `Test=72,454`) |

- **Collision Taxonomy Certification** *(retracts legacy "48.2% Benign" claim)*:
  - **96.49% of conflicting hashes (`8,433 / 8,740`, spanning `26,697 / 28,608 = 93.32%` collision flows)** are protocol-identical **DoS vs. DDoS volumetric floods** (`TCP`, `SYN`, `UDP`, `SynonymousIP`).
  - Only **41 hashes (`0.47%`) / 621 rows (`0.11%` of dataset)** involve `BENIGN` (mainly ARP vs. `MITM-ARPSPOOFING`).

### 2.2 Random-Split 80/20 Leakage Simulation Bounds
- **Unique Stratum (`N = 501,098`, Test `100,220`)**: Seed 42 leaked **`3,071` of `3,773`** test collision rows (**`81.39%`** collision leakage; **`3.06%`** total test leakage). **5-Seed Mean (42, 0, 123, 999, 2024)**: **`82.58% ± 1.05%`** collision leakage (**`3.03% ± 0.03%`** total test leakage). *(Retracts idealized "100%" phrasing.)*
- **Full Un-deduplicated Sample (`N = 529,562`, Test `105,913`, Seed 42)**: **`10,738`** test rows (`5,357` collision + `5,381` consistent) matched training coordinates (**`10.14%`** total test leakage; **`93.00%`** collision leakage).

---

## 3. Cross-Capture Macro-F1 Paradox & Weighting Reconciliation

- **Cross-Capture `fine_grained_lightgbm` Macro-F1 Shift (`0.6145 -> 0.6498`, `+0.0353`)**:
  - Proven to stem from **class prevalence shift** (capping classes at `<=2,000` in `Merged26–40` boosted rare attack support **`3.2×–5.2×`**: e.g., `BROWSERHIJACKING` `294 -> 1,320`, Precision `19.67% -> 51.18%`, F1 `0.2870 -> 0.5101`; `SQLINJECTION` `254 -> 1,233`, Precision `4.72% -> 23.16%`, F1 `0.0834 -> 0.2846`).
  - Claims of a "constant false-positive rate" are **retracted**: overall error rate *rose* (`27.35% -> 30.36%`), Global Acc *fell* (`72.65% -> 69.64%`), MCC *fell* (`0.7170 -> 0.6861`), Weighted-F1 *fell* (`0.7349 -> 0.6989`), and Binary LGBM Macro-F1 *fell* (`0.7637 -> 0.7034`).
- **Disentangled Unweighted Ablation Baselines (`fine_grained_lightgbm`, Balanced -> Unweighted)**:
  - **Validation (`75,172`)**: Acc `72.45% -> 74.33%` (`+1.88%`), Tier 4 Web Macro-Recall `36.67% -> 18.53%` (`-49.5%`), `SQLINJECTION` `34.13% -> 5.56%` (`-83.72%`), `XSS` `27.22% -> 7.10%` (`-73.91%`, fixing Stage 8A's swapped `XSS`/`BrowserHijacking` label).
  - **Internal Test (`75,105`)**: Acc `72.65% -> 74.45%` (`+1.80%`), Tier 4 Web Recall `34.63% -> 16.81%` (`-51.46%`), `SQLINJECTION` `35.83% -> 7.48%` (`-79.12%`), `XSS` `20.22% -> 3.83%` (`-81.08%`), `RECON-PINGSWEEP` `26.40% -> 4.80%` (`-81.82%`), Attacks->Benign FNs `1,517 -> 3,418` (`+125.31%`).

---

## 4. Reconciled Host-Side CPU Profiling & Manuscript Calibration (Stages 8B–8D)

All table rows in `paper.typ` / `main.tex` were verified cell-by-cell against `centralized_experiment_ledger.csv` (Intel Core i5-1135G7 host CPU; sequential unbatched single-flow P50/P95 vs. 10k-row in-memory batch throughput; prohibiting wire-speed line-rate claims):

| Granularity | Model | P50 (us) | P95 (us) | Batch (flows/s) | Peak RSS (MB) | Disk Size |
|---|---|---:|---:|---:|---:|---:|
| **Binary (2)** | SGD LogReg / MLP | 362.1 / 247.1 | 518.2 / 557.3 | 8,325,000 / 163,500 | 481.7 / 650.9 | 1.3 KB / 63.3 KB |
| **Binary (2)** | LGBM / XGB / RF | 1,983.3 / 1,575.3 / 70,883.1 | 2,227.0 / 2,130.2 / 123,496.3 | 938,400 / 1,424,300 / 62,400 | 494.7 / 497.5 / 498.9 | 0.36 / 0.26 / 12.35 MB |
| **Category (8)** | SGD LogReg / MLP | 143.2 / 613.8 | 247.5 / 1,372.0 | 8,007,700 / 123,600 | 484.7 / 638.6 | 2.5 KB / 64.9 KB |
| **Category (8)** | LGBM / XGB / RF | 1,409.8 / 2,244.0 / 22,467.5 | 1,852.1 / 3,235.0 / 41,428.5 | 92,400 / 264,800 / 330,300 | 484.8 / 483.6 / 526.9 | 2.81 / 2.11 / 36.47 MB |
| **Fine (34)** | SGD LogReg / MLP | 189.7 / 942.3 | 495.9 / 1,405.9 | 2,901,500 / 74,700 | 482.8 / 638.9 | 7.1 KB / 71.8 KB |
| **Fine (34)** | LGBM / XGB / RF | 3,504.4 / 3,732.5 / 31,296.1 | 4,882.8 / 6,092.7 / 46,911.1 | 10,990 / 47,560 / 157,490 | 493.8 / 501.8 / 572.6 | 11.88 / 8.67 / 95.65 MB |

- **Manuscript Packaging Verified**: 9-section dual-format manuscripts ([`../paper/paper.typ`](../paper/paper.typ), [`../paper/main.tex`](../paper/main.tex)), 23 BibTeX entries (`references.bib`, 100% parity with in-text citations across all 14 reviewed studies + ML toolkits), 6 regenerated 300 DPI figures (`fig1–fig6`), and Typst-compiled 11-page PDF (`research_paper.pdf`, 3.90 MB) — with hyperbolic/GBDT-immunity/microcontroller overclaims scrubbed under the 3-tier claim ledger.
