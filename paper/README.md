# Capstone Research Paper Package

This directory contains the academic research paper for the Capstone Project:
**"Hierarchical Multi-Tier Network Intrusion Detection and Edge Latency Optimization on the CICIoT2023 Benchmark"**

## Files Included

1. **`research_paper.pdf`**:
   The compiled, publication-quality research paper PDF formatted in IEEE two-column conference layout. Includes abstract, mathematical formulations, master benchmark tables, minority attack defense deep-dives, and embedded high-resolution figures.

2. **`main.tex`**:
   The complete, standard LaTeX source document formatted with the `IEEEtran` conference document class. Ready for direct submission or compilation in:
   - [Overleaf](https://www.overleaf.com) (simply upload this `paper/` directory as a ZIP)
   - Local TeX distributions (TeX Live, MiKTeX, MacTeX) via `pdflatex main.tex`

3. **`references.bib`**:
   Complete BibTeX citation database containing IEEE-formatted references for the CICIoT2023 dataset, XGBoost, LightGBM, Random Forest, PyTorch, and foundational NIDS surveys.

4. **`paper.typ`**:
   The source document formatted using Typst, enabling sub-second local compilation to PDF without requiring a multi-gigabyte TeX Live installation.

5. **`figures/`**:
   Contains all high-resolution figures embedded into the paper:
   - `capstone_master_synthesis.png`: 4-panel master synthesis (Accuracy progression, Macro F1 resilience, Latency vs F1 Pareto frontier, and line-rate network throughput).
   - `feature_importance_ranking.png`: Consensus tree feature importance and cumulative information retained curve.
   - `feature_reduction_comparison.png`: Accuracy retention vs inference latency speedup.
   - `attack34_cm_xgboost.png`: 34x34 fine-grained confusion matrix.

## How to Re-Compile

### Option A: Via Local Python Environment (Typst)
```bash
python -c "import typst; typst.compile('paper/paper.typ', output='paper/research_paper.pdf')"
```

### Option B: Via Standard LaTeX / Overleaf
```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
Or upload the entire `paper/` folder directly to Overleaf.
