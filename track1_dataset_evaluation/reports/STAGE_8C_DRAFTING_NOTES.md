# Stage 8C: Research Manuscript Drafting Notes

**Project Title**: Empirical Evaluation and Methodological Auditing of Machine Learning Baselines for IoT Network Intrusion Detection on the CICIoT2023 Benchmark  
**Deliverable Type**: Stage 8C Working Notes and Drafting Inventory  
**Target Manuscript Files**:
- LaTeX Master: [`paper/main.tex`](../paper/main.tex)
- Typst Master: [`paper/paper.typ`](../paper/paper.typ)
- Compiled Publication PDF: [`paper/research_paper.pdf`](../paper/research_paper.pdf)
- Bibliographic Master: [`paper/references.bib`](../paper/references.bib)
- Reproducible Figures Generator: [`scripts/generate_paper_figures.py`](../scripts/generate_paper_figures.py)
- Publication Vector Figures: [`paper/figures/`](../paper/figures/)
**Authoritative Blueprint**: [`reports/STAGE_8B_MANUSCRIPT_PLAN.md`](STAGE_8B_MANUSCRIPT_PLAN.md)  
**Ground-Truth Foundation**: [`reports/STAGE_8A2_AUDIT_VERIFICATION.md`](STAGE_8A2_AUDIT_VERIFICATION.md)  

---

## 1. Executive Drafting Completion Summary

Stage 8C has been executed to completion in a continuous, end-to-end workflow without intermediate halts or alterations to upstream datasets, models, or empirical result ledgers. 

The primary manuscript has been fully authored in both **LaTeX** (`paper/main.tex`) and **Typst** (`paper/paper.typ`), following standard academic formats (*Computers & Security* / IEEE Transactions style). Both manuscript versions are equivalent in narrative structure, mathematical formulation, numerical constants, table layouts, and visual figure placements.

The Typst manuscript compiles directly to a high-fidelity, 11-page publication-ready PDF (`paper/research_paper.pdf`) via the Python `typst` engine.

---

## 2. Deliverables Inventory

| Artifact Path | Format | Status | Description |
|---|---|---|---|
| `paper/references.bib` | BibTeX | Complete | Contains full bibliographic entries for all 14 surveyed peer-reviewed studies (2021--2026) and algorithm foundational citations. |
| `scripts/generate_paper_figures.py` | Python 3 | Complete | Standalone, fully reproducible script generating Figures 1--6 at 300 DPI vector-like resolution from repository ledgers. |
| `paper/figures/fig1_partitioning_protocol.png` | PNG (300 DPI) | Generated | High-level architectural flowchart of the Zero-Leakage Atomic Hash-Grouped Partitioning Protocol. |
| `paper/figures/fig2_accuracy_vs_macro_f1.png` | PNG (300 DPI) | Generated | Scatter plot of In-Distribution Test Generalization (Macro-F1 vs. Accuracy) across 21 models and 3 granularities. |
| `paper/figures/fig3_minority_recall_collapse.png` | PNG (300 DPI) | Generated | Horizontal bar chart illustrating the "Accuracy Paradox": minority attack recall collapse under unweighted training. |
| `paper/figures/fig4_threat_tier_cross_capture.png` | PNG (300 DPI) | Generated | Grouped bar chart comparing Internal Test vs. Cross-Capture Holdout F1 scores across 5 threat tiers. |
| `paper/figures/fig5_latency_pareto_frontier.png` | PNG (300 DPI) | Generated | Log-scale host-side P50 latency vs. Macro-F1 Pareto frontier across 15 primary baseline configurations. |
| `paper/figures/fig6_error_topology_matrix.png` | PNG (300 DPI) | Generated | 34-class fine-grained confusion matrix highlighting intra-protocol flood collisions and benign handshake overlap. |
| `paper/main.tex` | LaTeX (IEEEtran) | Complete | Complete, full-length academic paper in LaTeX format with 9 comprehensive sections, 9 tables, and 6 figures. |
| `paper/paper.typ` | Typst | Complete | Complete, full-length academic paper in Typst format mirroring `main.tex`. |
| `paper/research_paper.pdf` | PDF (11 pages) | Compiled | Full compiled manuscript rendering text, math, tables, vector figures, and IEEE-style bibliography cleanly. |

---

## 3. Structural Mapping & Section Plan Adherence

The drafted manuscripts rigorously adhere to the 9-section structure planned in Stage 8B:

1. **Title & Abstract**: Defensible paper title (*Unmasking Data Leakage and Temporal Fragility in IoT Network Intrusion Detection: An Empirical Audit and Benchmark on CICIoT2023*), structured author blocks, and a ~250-word abstract synthesizing findings with exact certified numbers.
2. **Section 1: Introduction**:
   - Modern IoT threat landscape and edge hardware constraints.
   - The "Accuracy Paradox" in contemporary benchmarks.
   - Core methodological flaws in prior tabular pipelines (leakage, class imbalance masking, lack of temporal evaluation).
   - Central RQ and five Sub-RQs (RQ1--RQ5).
   - Four consolidated research contributions explicitly stated.
3. **Section 2: Related Work & Research Gap Analysis**:
   - Thematic audit of recent literature across 14 peer-reviewed studies (Themes A--D).
   - Comparative literature matrix (Table 1) evaluating duplication/collision audits, temporal testing, models, and limitations.
   - The narrowly supported research gap (zero collision audits, zero temporal cross-capture tests across the 14 surveyed studies).
4. **Section 3: Dataset Integrity, Collisions, and Leakage Analysis**:
   - CICIoT2023 39-feature space topology.
   - Granular census of duplication (Table 2: 529,562 full sample vs. 501,098 unique stratum).
   - Conflicting label collision taxonomy (Table 3: 8,740 hashes spanning 28,608 / 18,416 flows; 48.2% collision entanglement with Benign).
   - Random-split leakage simulation (Table 4: 81.39% test collision leakage; 10.14% full-sample test leakage).
5. **Section 4: Leakage-Controlled Methodology**:
   - Atomic SHA-256 group-hash partitioning protocol ($\mathcal{X}_{\text{train}} \cap \mathcal{X}_{\text{test}} = \emptyset$).
   - Partition formulations: Strategy 1 (Collision-Contaminated) vs. Strategy 2 (Collision-Purged).
   - Temporal cross-capture clean holdout design ($N = 57,280$).
   - Preprocessor isolation ($\boldsymbol{\mu}, \boldsymbol{\sigma}^2$ strictly fitted on train).
   - Model families (5 architectures, 21 configurations) and inverse-frequency cost-sensitive loss weighting formulation.
6. **Section 5: Experimental Benchmark Evaluation**:
   - Master benchmark evaluation across 21 models (Table 5).
   - In-distribution generalization stability ($|\Delta_{\text{gen}}| \le 0.0079$).
   - Ambiguity filtering mechanics (Table 6: Common-Clean $N=72,390$ vs. Ambiguous $N=2,782$; observed collision accuracy of ~41.73%).
   - Architectural trade-offs between GBDTs, Random Forest, linear SGD, and compute-constrained MLP.
7. **Section 6: Security Trade-Offs, Temporal Robustness, and Systems Profiling**:
   - The Accuracy Paradox and minority attack recall collapse under unweighted training (Table 7, Figure 3).
   - Temporal cross-capture degradation (Table 8, Figure 4: Binary LightGBM Macro-F1 drop from 0.7637 to 0.7034).
   - Multiclass prevalence shift deconstruction on rare web attack classes.
   - Error topology and confusion matrix analysis (Figure 6).
   - Host-side computational profiling and Pareto frontier (Table 9, Figure 5: P50 latency, throughput, RSS, and disk sizes).
8. **Section 7: Discussion and Practical Implications**:
   - Re-evaluating published performance on CICIoT2023.
   - Four operational recommendations (hash-grouped partitioning, cost-sensitive objectives, multi-tier cascades, selective DPI).
9. **Section 8: Threats to Validity and Limitations**:
   - Dataset/testbed scope, packet payload inaccessibility, 5-epoch neural budget, host CPU profiling boundaries, literature scope.
10. **Section 9: Conclusion and Future Work**:
    - Bullet-proof summary of findings, leak-free benchmark established, open avenues for physical edge microcontroller deployment.

---

## 4. Verification of Ground-Truth Constants

All numerical values embedded in text, equations, and tables have been verified against the certified Stage 8A.2 ground truths:

- [x] **Sample Sizes**: Full sample ($N = 529,562$), Unique Stratum ($N = 501,098$), Exact Duplicates ($28,464$ / $5.38\%$), Repeated Feature Flows ($59,780$ / $11.29\%$).
- [x] **Collisions**: Conflicting Hashes ($8,740$), Collision Flows in Full Sample ($28,608$ / $5.40\%$), Collision Flows in Unique Stratum ($18,416$ / $3.68\%$), Strategy 2 Purged Population ($482,682$).
- [x] **Random-Split Leakage**: Unique Stratum Seed 42 Collision Leakage ($81.39\%$), 5-Seed Collision Leakage ($82.58\% \pm 1.05\%$), Full Sample Test Leakage ($10.14\%$).
- [x] **Primary Benchmark Metrics (Internal Test $N=75,105$)**:
  - Binary RF: Accuracy $89.61\%$, Macro-F1 $0.7690$, MCC $0.6057$.
  - Binary LightGBM: Accuracy $89.15\%$, Macro-F1 $0.7637$, MCC $0.6007$.
  - Category LightGBM: Accuracy $78.93\%$, Macro-F1 $0.6894$, MCC $0.7468$.
  - Fine-Grained LightGBM: Accuracy $72.65\%$, Macro-F1 $0.6145$, MCC $0.7170$, Macro-AP $0.6619$, ROC-AUC $0.9811$.
- [x] **Generalization Stability**: $|\Delta_{\text{gen}}| \le 0.0079$ across all 21 models.
- [x] **Ambiguity Routing**: Common-Clean Gain ($\Delta = +0.0022$ Val Macro-F1, $+0.0013$ Cross Macro-F1), Ambiguous Accuracy ($41.73\%$).
- [x] **Class-Weighting Ablation**:
  - Test Accuracy: $72.65\% \to 74.45\%$ ($+1.80\%$).
  - SQL Injection Recall: $35.83\% \to 7.48\%$ ($-79.12\%$ relative drop).
  - Recon-PingSweep Recall: $26.40\% \to 4.80\%$ ($-81.82\%$ relative drop).
  - XSS Recall: $20.22\% \to 3.83\%$ ($-81.08\%$ relative drop).
  - False Negatives to Benign: $1,517 \to 3,418$ flows ($+125.31\%$).
- [x] **Temporal Cross-Capture**:
  - Binary LightGBM Macro-F1: $0.7637 \to 0.7034$ ($-0.0603$).
  - Binary LightGBM Accuracy: $89.15\% \to 83.48\%$ ($-5.67\%$).
- [x] **Computational Profiling (Host CPU)**:
  - Single-flow P50 latency: SGD ($143\,\mu\text{s}$), MLP ($247\,\mu\text{s}$), LightGBM ($1,409\,\mu\text{s}$), XGBoost ($1,575\,\mu\text{s}$), RF ($22,467\,\mu\text{s}$).
  - Peak RSS: MLP ($650.9\text{ MB}$), RF ($498.9\text{--}572.6\text{ MB}$), LightGBM ($494.7\text{ MB}$), XGBoost ($482.1\text{ MB}$), SGD ($312.4\text{ MB}$).

---

## 5. Three-Tier Claim Ledger Adherence

- **Tier 1 (Supported Findings)**: All claims regarding leakage rates, collision prevalence, benchmark accuracies, minority recall collapse, and temporal degradation are accompanied by exact empirical metrics and population denominators.
- **Tier 2 (Qualified Interpretations)**: Explanations regarding IP stripping, packet header dynamics, and application payload blindness are expressly characterized as hypotheses or structural domain properties.
- **Tier 3 (Prohibited Claims Strictly Excluded)**:
  - *No Bayes Error Claim*: The ~41.73% ambiguous accuracy is described strictly as empirical observed accuracy on non-injective coordinates.
  - *No Hardware Line-Rate or Microcontroller Claim*: Latency metrics are explicitly presented as single-thread host CPU inference on an Intel Core i5.
  - *No Sweeping Generalization*: Literature statements are bounded strictly to the 14 audited peer-reviewed studies.
  - *No Unqualified Tree Superiority Claim*: Tree dominance is discussed specifically within tabular flow features under the evaluated CPU compute budgets.

---

## 6. Compilation Status and Quality Audit

1. **Typst Engine**:
   - Successfully compiled `paper/paper.typ` into `paper/research_paper.pdf` using Python `typst` library.
   - Verified PDF output: 11 pages, clean typography, two-column layout, embedded vector figures, balanced tables, and IEEE-style bibliography.
2. **LaTeX Engine**:
   - `paper/main.tex` syntax and structure audited:
     - 100% of citation keys (`\cite{...}`) exist in `paper/references.bib` (19 citations verified).
     - 100% of figure paths exist in `paper/figures/`.
     - 100% of LaTeX environments (`table`, `table*`, `figure`, `enumerate`, `itemize`, `tabular`, `equation`) are balanced and properly closed.
   - Note: Standalone external CLI compilers (`pdflatex`, `latexmk`, `typst.exe`) are not installed on the host OS system PATH. The manuscript has been compiled and validated directly via the Python `typst` package into `paper/research_paper.pdf`.

---

## 7. Action Items for Stage 8D Final Polish & Review

1. **Author Affiliations & Footnotes**: Finalize student registration numbers and supervisor contact info if updated formatting guidelines require camera-ready formatting.
2. **Journal Style Template Adaptation**: If submitting to *Computers & Security* (Elsevier), convert from the IEEEtran layout to the Elsevier `elsarticle` LaTeX class.
3. **Supplementary Material Generation**: Package the 21 model evaluation CSVs and the collision analysis script into an archival Zenodo/GitHub release bundle.
4. **Final Proofreading**: Conduct automated spelling and grammar checks across all 9 sections.

---
*Stage 8C Drafting Complete.*  
*Ready for Stage 8D: Final Review, Manuscript Polishing, and Submission Packaging.*
