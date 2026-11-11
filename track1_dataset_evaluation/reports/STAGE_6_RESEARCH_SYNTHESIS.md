# Stage 6: Research Findings, Z-Score Preprocessing Proof & Claim Synthesis

> Synthesizes Stages 1–5 findings, preprocessor verification (`LeakFreePreprocessor`), methodological limitations, and calibrated scientific claims prior to literature benchmarking (Stage 7) and manuscript drafting (Stage 8). Full test numbers reside in [`STAGE_4_EVALUATION_REPORT.md`](STAGE_4_EVALUATION_REPORT.md) and [`../results/tables/stage4_test_results.csv`](../results/tables/stage4_test_results.csv).

---

## 1. Research Questions Addressed

- **RQ1 (Integrity)**: How prevalent are duplicate vectors and conflicting label collisions in CICIoT2023, and can atomic hash partitioning prevent inter-partition feature overlap?
- **RQ2 (Baselines by Granularity)**: How do SGD LogReg, PyTorch MLP, Random Forest, LightGBM, and XGBoost compare across Binary, 8-Class Category, and 34-Class tasks when leakage is eliminated?
- **RQ3 (Ambiguity & Collision Purging)**: How does training with conflicting collision hashes (Strategy 1) vs. purging them (Strategy 2) affect internal test and cross-capture evaluation?
- **RQ4 (Class Imbalance Trade-Offs)**: How does inverse-frequency class weighting trade aggregate accuracy and Benign FNs against minority attack recall?
- **RQ5 (Temporal Session Stability)**: How stable are models trained on `Merged01–25` when evaluated on later session captures (`Merged26–40`) from the same IoT testbed?

---

## 2. Critical Stage 4 Verification: Preprocessor Z-Score Coupling

Inspection of [`../src/run_stage3.py`](../src/run_stage3.py) verified `scale_features=True` across all 21 models—including decision-tree ensembles. Evaluating `fine_grained_lightgbm` with `preprocessor_strategy1.joblib` reproduced Stage 3 validation accuracy (`72.46%`) at `Δ = 0.00000`, whereas feeding unscaled raw inputs collapsed tree accuracy to `7.31%` because split thresholds were learned in standardized $z$-score space. Both serialized preprocessors must always accompany inference.

---

## 3. Core Empirical Findings & Methodological Contributions

1. **Val-to-Test Consistency (`|Δ_gen| <= 0.0079`)**: Across all 21 models, Internal Test Macro-F1 closely tracked Validation (e.g., `fine_grained_lightgbm`: `0.6142 -> 0.6145`; `binary_random_forest` leads Binary at `89.61%` / `0.7690` Macro-F1; `category_lightgbm` leads 8-Class at `78.93%` / `0.6894`; `fine_grained_lightgbm` leads balanced 34-Class at `72.65%` / `0.6145`).
2. **Collision Purging Has Marginal Effect (`Δ = +0.0022` clean Val / `+0.0013` Cross Macro-F1)**: Purging all `8,740` collision hashes changed Internal Test Macro-F1 from `0.6145 -> 0.6140` (`-0.0005`) and Cross-Capture Macro-F1 from `0.6498 -> 0.6511` (`+0.0013`), while isolated validation performance on the `2,782` ambiguous rows hovered at `~41.73%`.
3. **Unweighted Optimization Induces Minority Starvation & +125.3% Benign FN Surge**: Dropping balanced class weights raised LightGBM Test Accuracy (`72.65% -> 74.45%`) while slashing Tier 4 Web Macro-Recall from `34.63% -> 16.81%` (`SQLINJECTION`: `35.83% -> 7.48%`, `XSS`: `20.22% -> 3.83%`, `UPLOADING_ATTACK`: `23.33% -> 6.67%`) and raising Attacks->Benign misses from `1,517` (`2.20%`) to `3,418` (`4.95%`).
4. **Distinct Methodological Contributions vs. Standard Practice**: Atomic feature-hash partitioning + common clean `72,390`-row validation design + full collision census + dual same-testbed cross-capture evaluation constitute the core contributions; preprocessor serialization and read-only single-pass evaluation are standard engineering hygiene.

---

## 4. Calibrated Claim Taxonomy & Explicit Limitations

- **Demonstrated Findings**: Zero overlap under group-hash partitioning; `|Δ_gen| <= 0.0079` internal split stability; task-specific winners (RF Binary, LightGBM Category/Fine); unweighted `+125.3%` Benign FN surge and Web recall collapse; `~41.7%` observed accuracy on ambiguous subset; marginal Strategy 2 deltas.
- **Supported Interpretations**: Intra-tier flood collisions (`DDoS` vs `DoS`, `83.3%` of top confusion pairs) stem from identical packet header profiles without source-IP entropy; low-rate scans (`RECON-OSSCAN`, `21.06% -> BENIGN`) resemble routine resets.
- **Explicit Boundaries**: Single 105-device laboratory testbed; `cross_capture_test_clean.csv` evaluates same-testbed session shift (`Merged26–40`, capped `2,000`/attack) rather than external networks; 39 payload-blind statistical features cap Web exploit separability; PyTorch MLP reflects a 5-epoch CPU budget; host profiling reflects user-space workstation CPU, not hardware line-rate eBPF/DPDK.
