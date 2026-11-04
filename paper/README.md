# Academic Research Manuscript Package (`paper/`)

This directory contains the academic research manuscript, typesetting sources, publication figures, and compiled PDF deliverable for the capstone project:

> **"Empirical Evaluation of Data Leakage, Class Imbalance, and Temporal Fragility in IoT Network Intrusion Detection on CICIoT2023"**  
> *Authors:* Priyajit Biswal, Rohit K Manoj, Rishab Rajeev  
> *Affiliation:* Department of Computer Science & Engineering, Shiv Nadar University Chennai, India  
> *Faculty In Charge:* Dr. Vegesna S M Srinivasavarma  

---

## 1. Package Contents

| File / Directory | Format | Scope & Purpose |
|---|---|---|
| **[`research_paper.pdf`](research_paper.pdf)** | PDF (11 Pages) | The compiled publication-quality research paper PDF (3.90 MB) formatted in academic two-column layout. Includes abstract, mathematical formulations, 9 master tables, 6 figures, and IEEE-style references. |
| **[`main.tex`](main.tex)** | LaTeX (IEEEtran) | Complete, standard LaTeX source document formatted with the `IEEEtran` document class. Syntax-verified with 100% environment balance and 19 resolved citations. Ready for direct upload to Overleaf or local TeXLive compilation. |
| **[`paper.typ`](paper.typ)** | Typst | Modern Typst source document equivalent in content, math, tables, figures, and references to `main.tex`. Enables rapid, reproducible local compilation. |
| **[`references.bib`](references.bib)** | BibTeX | Comprehensive BibTeX database containing 23 verified entries, including all 14 peer-reviewed IoT NIDS studies audited in Stage 7 and foundational algorithmic citations (XGBoost, LightGBM, Random Forest, PyTorch, Scikit-Learn). |
| **[`figures/`](figures/)** | PNG (300 DPI) | High-resolution publication figures generated from repository experimental ledgers: |
| &emsp; `fig1_partitioning_protocol.png` | PNG | Schematic of the Zero-Leakage Atomic Hash-Grouped Partitioning Protocol. |
| &emsp; `fig2_accuracy_vs_macro_f1.png` | PNG | In-distribution test generalization scatter (Macro-F1 vs. Accuracy) across 21 models. |
| &emsp; `fig3_minority_recall_collapse.png` | PNG | The Accuracy Paradox: minority attack recall disparity under unweighted training. |
| &emsp; `fig4_threat_tier_cross_capture.png` | PNG | Temporal cross-capture generalization across threat tiers for fine-grained LightGBM. |
| &emsp; `fig5_latency_pareto_frontier.png` | PNG | Host-side latency vs. diagnostic Macro-F1 Pareto curve with reconciled P50 values. |
| &emsp; `fig6_error_topology_matrix.png` | PNG | 34-class fine-grained confusion matrix highlighting intra-flood and benign handshake overlap. |

---

## 2. Compilation Instructions

### Option A: Via Python Typst Engine (Recommended / Local)
To compile `paper.typ` into `research_paper.pdf` locally without requiring a full TeXLive installation:
```powershell
.\.venv\Scripts\python.exe -c "import typst; typst.compile('paper/paper.typ', output='paper/research_paper.pdf')"
```

### Option B: Via Standard LaTeX / Overleaf
To compile `main.tex` into PDF using a local TeXLive or MiKTeX installation:
```powershell
pdflatex paper/main.tex
bibtex paper/main
pdflatex paper/main.tex
pdflatex paper/main.tex
```
*Alternatively, the entire `paper/` folder can be uploaded directly as a ZIP archive to [Overleaf](https://www.overleaf.com).*

---

## 3. Regenerating Publication Figures

All six figures embedded in the paper can be deterministically regenerated from [`results_v2/tables/`](../results_v2/tables/) at 300 DPI:
```powershell
.\.venv\Scripts\python.exe scripts/generate_paper_figures.py
```
