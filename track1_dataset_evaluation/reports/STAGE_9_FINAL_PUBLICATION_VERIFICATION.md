# Stage 9: Final Publication Verification, PDF Audit & Approval

> Authoritative Stage 9 sign-off for [`../paper/research_paper.pdf`](../paper/research_paper.pdf) (11 pages, 3.90 MB, IEEE two-column Typst build) and [`../paper/main.tex`](../paper/main.tex). Resolves all 10 manuscript review checks against certified Stage 4 / Stage 8 numerical ground truths.

---

## 1. Final Assessment & Resolutions Across 10 PDF Review Checks

**Verdict**: **`READY FOR FINAL APPROVAL`** (Typst 0.12.0 exit code `0`, zero unresolved citations, zero overflow/clipping across all 11 rendered pages; `main.tex` AST-verified with all environments balanced for TeXLive/Overleaf).

| # | Review Check | Certified Verification & Applied Manuscript Resolution |
|---|---|---|
| **1** | **Overlap vs. Memorization** | Qualified `81.39%` test collision overlap as creating an *empirical opportunity* for coordinate memorization rather than definitive proof of classifier strategy. |
| **2** | **Collision Taxonomy** | Retracted legacy "48.2% Benign" text; replaced with certified `96.49%` hashes (`26,697` flows = `93.32%`) DDoS/DoS protocol twins vs. `41` hashes (`0.47%` / `621` rows) Benign/ARP. |
| **3** | **Cross-Capture Provenance** | Documented `Merged26–40` (`N=57,280`, `57,148` unique vectors, `0` float-tuple overlap via `verify_partitions.py`) explicitly as *same-testbed cross-session stability* (timestamps absent from CSVs). |
| **4** | **Class Prevalence Denominators** | Cited exact `sample_stratified.csv` shares (`N=529,562`): DoS/DDoS = `56.02%` (`296,655`), Web/app = `1.51%` (`8,004`: `UPLOADING_ATTACK` `0.079%` [416], `XSS` `0.234%` [1,239], `SQLINJECTION` `0.324%` [1,718]). |
| **5** | **Cross-Capture Macro-F1 Shift** | Attributed `0.6145 -> 0.6498` jump partly to `3.2×–5.2×` rare-class support inflation while noting Accuracy (`-3.01%`), MCC (`-0.0309`), and Binary F1 (`-0.0603`) fell. |
| **6** | **Benign F1 Discrepancy** | Explicitly separated Binary LGBM Benign F1 (`0.5899 -> 0.5061`) from Fine-Grained 34-class LGBM Benign F1 (`0.6884 -> 0.6693`). |
| **7** | **DoS/DDoS Collision Cause** | Framed IP/timestamp removal during CICIoT2023 feature extraction as a *hypothesized contributing factor* eliminating source-IP entropy. |
| **8** | **Throughput vs. Single Latency** | Strictly separated sequential single-flow latency (500 warmup, 5×2,000 unbatched calls) from vectorized 10k-row RAM batch throughput, disallowing wire-speed/line-rate equivalence. |
| **9** | **Abstract Model Taxonomy** | Distinguished bagging ensemble (**Random Forest** Binary `89.61%`, Macro-F1 `0.7690`) from gradient-boosted trees (**LightGBM** Category `78.93%` / `0.6894`, Fine `72.65%` / `0.6145`). |
| **10** | **Literature Claim Bounds** | Restricted gap claims strictly to the 14 surveyed studies without generalizing leakage to all >99% papers globally. |

---

## 2. Certified Publication Verification Checklist

- **Typst & LaTeX**: Compiled cleanly to 11 balanced pages (`paper/research_paper.pdf`); 23 BibTeX entries resolve 100% of in-text citations (`21` cited keys across `paper.typ` & `main.tex`); Figures 1–6 regenerated at 300 DPI (`scripts/generate_paper_figures.py`).
- **Remaining Explicit Boundaries**: CICIoT2023 single testbed only (no BoT-IoT / ToN_IoT external transfer); 39 payload-blind statistical features cap Web-exploit precision; CSV rows omit wall-clock timestamps; workstation Intel i5-1135G7 profiling excludes NIC eBPF/DPDK parsing and embedded MCU limits; 14-paper literature scope.
