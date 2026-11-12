"""
Experiment 3 Runner: Zero-Flood Rare Non-Flood Specialist + Validation Policy Tuning.

Trains:
  - Stage A: Binary Rare-8 vs Safe/Common Non-Flood detector (161,533 Zero-Flood Train rows).
  - Stage B: 8-class Rare Non-Flood classifier (9,108 Rare Train rows).
Evaluates components & sweeps hybrid integration policies strictly on Validation (75,172 rows)
for both ORIG34 and OP29 taxonomies, comparing Naive tau=0.50 override vs Validation-gated policy.
Internal Test & Cross-Capture are NEVER loaded or inspected here.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from track1_dataset_evaluation.src.config import THIRTY_FOUR_CLASSES
from track2_ids_engineering.hierarchical_ids.src.data_utils import (
    FLOOD_CLASSES,
    HIER_MODELS_DIR,
    MINORITY_ATTACK_CLASSES,
    NON_FLOOD_CLASSES,
)
from track2_ids_engineering.minority_optimization.src.common import (
    CONFIGS_DIR,
    E4_LGBM_PARAMS,
    MODELS_DIR,
    OP29_BENIGN_IDX,
    OP29_CLASSES,
    OP29_FLOOD_CLASSES,
    OP29_MINORITY_CLASSES,
    OP29_NON_FLOOD_CLASSES,
    PROJ_34_TO_29,
    RESULTS_DIR,
    compute_power_sample_weights,
    load_all_partitions_read_only,
)
from track2_ids_engineering.minority_optimization.src.specialist import (
    BENIGN_34_IDX,
    RARE_8_IN_29_IDX,
    RARE_8_IN_34_IDX,
    analyze_routing_and_override_outcomes,
    apply_gated_hybrid_policy,
    build_zero_flood_specialist_train_pools,
)

TAU_RARE_GRID = [0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
TAU_CLS_GRID = [0.0, 0.25, 0.40, 0.55]
TAU_E4_KEEP_GRID = [0.65, 0.75, 0.85, 1.01]


def _fast_metrics_tax(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
    rare_idx: np.ndarray,
    benign_idx: int,
) -> Dict[str, float]:
    """Vectorized accuracy, macro-F1, minority-F1, minority-recall, Benign FPR, Attack->Benign FNR."""
    f1s = f1_score(y_true, y_pred, labels=np.arange(n_classes), average=None, zero_division=0)
    acc = float(np.mean(y_pred == y_true))
    macro_f1 = float(np.mean(f1s))
    min_f1 = float(np.mean(f1s[rare_idx]))

    # Minority recall
    rare_rec_list = []
    for cid in rare_idx:
        m = y_true == cid
        rare_rec_list.append(float(np.mean(y_pred[m] == cid)) if np.any(m) else 0.0)
    min_rec = float(np.mean(rare_rec_list))

    tb = y_true == benign_idx
    ta = ~tb
    pred_att = y_pred != benign_idx
    benign_fpr = float(np.mean(pred_att[tb])) if np.any(tb) else 0.0
    att_ben_fnr = float(np.mean(~pred_att[ta])) if np.any(ta) else 0.0

    return {
        "accuracy": round(acc, 5),
        "macro_f1": round(macro_f1, 5),
        "minority_f1": round(min_f1, 5),
        "minority_recall": round(min_rec, 5),
        "benign_fpr": round(benign_fpr, 5),
        "attack_to_benign_fnr": round(att_ben_fnr, 5),
    }


def main(force: bool = False) -> None:
    t0_all = time.perf_counter()
    print("=== EXPERIMENT 3: ZERO-FLOOD RARE SPECIALIST & HYBRID POLICY TUNING ===")
    splits = load_all_partitions_read_only(include_test=False, include_cross_capture=False)
    pools = build_zero_flood_specialist_train_pools(splits)

    spec_dir = MODELS_DIR / "exp3_specialist"
    spec_dir.mkdir(parents=True, exist_ok=True)

    # 1. Stage A Binary Detector Sweep (alpha in {0.0, 0.50, 1.00})
    print("\n[1/4] Training Stage A Binary Rare-8 Detectors (161,533 Zero-Flood Train rows, 0 Flood rows)...")
    stage_a_rows = []
    stage_a_models = {}
    for alpha, tag in [(0.0, "unweighted"), (0.50, "sqrt_a050"), (1.00, "balanced_a100")]:
        name = f"StageA_Bin_{tag}"
        ckpt = spec_dir / f"{name}.joblib"
        if ckpt.exists() and not force:
            clf = joblib.load(ckpt)
        else:
            t_s = time.perf_counter()
            sw = compute_power_sample_weights(pools["y_tr_stage_a"], alpha=alpha)
            clf = lgb.LGBMClassifier(**E4_LGBM_PARAMS)
            clf.fit(pools["X_tr_stage_a"], pools["y_tr_stage_a"], sample_weight=sw)
            joblib.dump(clf, ckpt, compress=3)
            print(f"  - Fitted {name} in {time.perf_counter() - t_s:.2f}s")
        stage_a_models[name] = (clf, ckpt, alpha)

        # Validation evaluation on (a) Zero-Flood Val rows and (b) All Val rows
        p_nf = clf.predict_proba(pools["X_val_nf_stage_a"])[:, 1]
        y_nf = pools["y_val_nf_stage_a"]
        p_all = clf.predict_proba(pools["X_val_all"])[:, 1]
        y_all = pools["y_val_all_stage_a"]

        row = {
            "component": "StageA_Binary",
            "model": name,
            "weight_alpha": alpha,
            "val_nf_pr_auc": round(float(average_precision_score(y_nf, p_nf)), 5),
            "val_nf_roc_auc": round(float(roc_auc_score(y_nf, p_nf)), 5),
            "val_nf_f1_at_05": round(float(f1_score(y_nf, p_nf >= 0.5)), 5),
            "val_nf_recall_at_05": round(float(np.mean((p_nf >= 0.5)[y_nf == 1])), 5),
            "val_nf_fpr_at_05": round(float(np.mean((p_nf >= 0.5)[y_nf == 0])), 5),
            "val_all_pr_auc": round(float(average_precision_score(y_all, p_all)), 5),
            "val_all_roc_auc": round(float(roc_auc_score(y_all, p_all)), 5),
            "val_all_fpr_floods_at_05": round(float(np.mean((p_all >= 0.5)[splits["val"]["is_flood"]])), 5),
            "size_mb": round(ckpt.stat().st_size / (1024 * 1024), 3),
        }
        stage_a_rows.append(row)
        print(
            f"    {name}: NF_PR-AUC={row['val_nf_pr_auc']:.5f} | NF_ROC-AUC={row['val_nf_roc_auc']:.5f} | "
            f"NF_F1@0.5={row['val_nf_f1_at_05']:.5f} | Flood_FP_if_unmasked@0.5={row['val_all_fpr_floods_at_05']:.5f}"
        )

    # Select best Stage A on Zero-Flood Validation PR-AUC
    best_a_row = sorted(stage_a_rows, key=lambda r: (r["val_nf_pr_auc"], r["val_nf_roc_auc"]), reverse=True)[0]
    best_a_name = best_a_row["model"]
    best_a_model, best_a_ckpt, _ = stage_a_models[best_a_name]
    print(f"  -> Selected Stage A on Validation: {best_a_name} (NF PR-AUC={best_a_row['val_nf_pr_auc']:.5f})")

    # 2. Stage B 8-Class Rare Classifier Sweep (alpha in {0.0, 0.50, 1.00})
    print("\n[2/4] Training Stage B 8-Class Rare Classifiers (9,108 Rare Train rows)...")
    stage_b_rows = []
    stage_b_models = {}
    for alpha, tag in [(0.0, "unweighted"), (0.50, "sqrt_a050"), (1.00, "balanced_a100")]:
        name = f"StageB_Rare8_{tag}"
        ckpt = spec_dir / f"{name}.joblib"
        if ckpt.exists() and not force:
            clf = joblib.load(ckpt)
        else:
            t_s = time.perf_counter()
            sw = compute_power_sample_weights(pools["y_tr_stage_b"], alpha=alpha)
            clf = lgb.LGBMClassifier(**E4_LGBM_PARAMS)
            clf.fit(pools["X_tr_stage_b"], pools["y_tr_stage_b"], sample_weight=sw)
            joblib.dump(clf, ckpt, compress=3)
            print(f"  - Fitted {name} in {time.perf_counter() - t_s:.2f}s")
        stage_b_models[name] = (clf, ckpt, alpha)

        yb_val = pools["y_val_stage_b"]
        pb_val = clf.predict(pools["X_val_stage_b"])
        row = {
            "component": "StageB_8Class_Oracle",
            "model": name,
            "weight_alpha": alpha,
            "val_rare_accuracy": round(float(np.mean(pb_val == yb_val)), 5),
            "val_rare_macro_f1": round(float(f1_score(yb_val, pb_val, average="macro", zero_division=0)), 5),
            "val_rare_mcc": round(float(matthews_corrcoef(yb_val, pb_val)), 5),
            "size_mb": round(ckpt.stat().st_size / (1024 * 1024), 3),
        }
        stage_b_rows.append(row)
        print(
            f"    {name}: OracleAcc={row['val_rare_accuracy']:.5f} | OracleMacroF1={row['val_rare_macro_f1']:.5f} | "
            f"MCC={row['val_rare_mcc']:.5f}"
        )

    best_b_row = sorted(stage_b_rows, key=lambda r: (r["val_rare_macro_f1"], r["val_rare_accuracy"]), reverse=True)[0]
    best_b_name = best_b_row["model"]
    best_b_model, best_b_ckpt, _ = stage_b_models[best_b_name]
    print(f"  -> Selected Stage B on Validation: {best_b_name} (Oracle Macro-F1={best_b_row['val_rare_macro_f1']:.5f})")

    pd.DataFrame(stage_a_rows + stage_b_rows).to_csv(RESULTS_DIR / "exp3_specialist_component_val.csv", index=False)

    # 3. Precompute Validation probabilities once for fast exact policy sweep
    print("\n[3/4] Running Validation Gating Policy Sweep (Naive vs Validated Thresholds)...")
    e4_ckpt = HIER_MODELS_DIR / "e4_lgbm_leaves63_sqrt.joblib"
    e4_model = joblib.load(e4_ckpt)
    X_val = splits["val"]["X"]
    p_e4_34_val = e4_model.predict_proba(X_val)
    p_e4_29_val = p_e4_34_val @ PROJ_34_TO_29
    p_a_val = best_a_model.predict_proba(X_val)[:, 1]
    p_b_val = best_b_model.predict_proba(X_val)

    tax_configs = [
        {
            "taxonomy": "ORIG34",
            "short": "34",
            "p_e4": p_e4_34_val,
            "y_val": splits["val"]["y_fine"],
            "n_classes": 34,
            "rare_idx": RARE_8_IN_34_IDX,
            "benign_idx": BENIGN_34_IDX,
        },
        {
            "taxonomy": "OP29",
            "short": "op29",
            "p_e4": p_e4_29_val,
            "y_val": splits["val"]["y_op29"],
            "n_classes": 29,
            "rare_idx": RARE_8_IN_29_IDX,
            "benign_idx": OP29_BENIGN_IDX,
        },
    ]

    selected_policies: Dict[str, Any] = {
        "stage_a_model": best_a_name,
        "stage_a_checkpoint": str(best_a_ckpt),
        "stage_b_model": best_b_name,
        "stage_b_checkpoint": str(best_b_ckpt),
        "policies": {},
    }

    for tc in tax_configs:
        tax = tc["taxonomy"]
        e4_pred = np.argmax(tc["p_e4"], axis=1)
        base_m = _fast_metrics_tax(tc["y_val"], e4_pred, tc["n_classes"], tc["rare_idx"], tc["benign_idx"])

        sweep_rows: List[Dict[str, Any]] = []

        # Build parameter grid + explicit Naive tau=0.50 override
        grid_tuples = [("Naive_tau050", 0.50, 0.0, 1.01)]
        for tr in TAU_RARE_GRID:
            for tc_thr in TAU_CLS_GRID:
                for te in TAU_E4_KEEP_GRID:
                    grid_tuples.append(("GridSearch", tr, tc_thr, te))

        for policy_kind, tr, tc_thr, te in grid_tuples:
            y_hyb, _, diag = apply_gated_hybrid_policy(
                p_e4_tax=tc["p_e4"],
                p_stage_a_rare=p_a_val,
                p_stage_b_8=p_b_val,
                taxonomy=tax,
                tau_rare=tr,
                tau_cls=tc_thr,
                tau_e4_keep=te,
            )
            m = _fast_metrics_tax(tc["y_val"], y_hyb, tc["n_classes"], tc["rare_idx"], tc["benign_idx"])
            rout = analyze_routing_and_override_outcomes(tc["y_val"], e4_pred, y_hyb, tax)
            row = {
                "taxonomy": tax,
                "policy_kind": policy_kind,
                "tau_rare": tr,
                "tau_cls": tc_thr,
                "tau_e4_keep": te,
                **m,
                "delta_macro_f1": round(m["macro_f1"] - base_m["macro_f1"], 5),
                "delta_minority_f1": round(m["minority_f1"] - base_m["minority_f1"], 5),
                "delta_accuracy": round(m["accuracy"] - base_m["accuracy"], 5),
                "delta_benign_fpr": round(m["benign_fpr"] - base_m["benign_fpr"], 5),
                **diag,
                **rout,
            }
            sweep_rows.append(row)

        df_sw = pd.DataFrame(sweep_rows)
        df_sw.to_csv(RESULTS_DIR / f"exp3_policy_val_sweep_{tc['short']}.csv", index=False)

        naive_row = df_sw[df_sw["policy_kind"] == "Naive_tau050"].iloc[0]

        # Validation policy selection under safety guardrails:
        # Guardrail: delta_benign_fpr <= +0.025, delta_accuracy >= -0.015, AND delta_macro_f1 >= 0 (or maximize composite inside guardrail)
        df_grid = df_sw[df_sw["policy_kind"] == "GridSearch"].copy()
        df_grid["composite_score"] = 0.5 * df_grid["macro_f1"] + 0.5 * df_grid["minority_f1"]
        guard = (
            (df_grid["delta_benign_fpr"] <= 0.025)
            & (df_grid["delta_accuracy"] >= -0.015)
        )
        pool = df_grid[guard] if guard.any() else df_grid
        best_gated = pool.sort_values(
            by=["composite_score", "macro_f1", "minority_f1", "accuracy"],
            ascending=False,
        ).iloc[0]

        selected_policies["policies"][tax] = {
            "naive": {
                "tau_rare": float(naive_row["tau_rare"]),
                "tau_cls": float(naive_row["tau_cls"]),
                "tau_e4_keep": float(naive_row["tau_e4_keep"]),
                "val_accuracy": float(naive_row["accuracy"]),
                "val_macro_f1": float(naive_row["macro_f1"]),
                "val_minority_f1": float(naive_row["minority_f1"]),
                "val_benign_fpr": float(naive_row["benign_fpr"]),
                "n_overrides": int(naive_row["overrides_total"]),
            },
            "gated": {
                "tau_rare": float(best_gated["tau_rare"]),
                "tau_cls": float(best_gated["tau_cls"]),
                "tau_e4_keep": float(best_gated["tau_e4_keep"]),
                "passed_guardrail": bool(guard.any()),
                "val_accuracy": float(best_gated["accuracy"]),
                "val_macro_f1": float(best_gated["macro_f1"]),
                "val_minority_f1": float(best_gated["minority_f1"]),
                "val_minority_recall": float(best_gated["minority_recall"]),
                "val_benign_fpr": float(best_gated["benign_fpr"]),
                "delta_macro_f1": float(best_gated["delta_macro_f1"]),
                "delta_minority_f1": float(best_gated["delta_minority_f1"]),
                "delta_benign_fpr": float(best_gated["delta_benign_fpr"]),
                "n_overrides": int(best_gated["overrides_total"]),
                "overrides_beneficial": int(best_gated["overrides_beneficial_wrong_to_right"]),
                "overrides_harmful": int(best_gated["overrides_harmful_right_to_wrong"]),
            },
        }
        print(
            f"\n  [{tax}] Baseline E4 Val: Acc={base_m['accuracy']:.5f}, MacroF1={base_m['macro_f1']:.5f}, "
            f"MinF1={base_m['minority_f1']:.5f}, BenFPR={base_m['benign_fpr']:.5f}"
        )
        print(
            f"  [{tax}] Naive tau=0.50: Acc={naive_row['accuracy']:.5f}, MacroF1={naive_row['macro_f1']:.5f}, "
            f"MinF1={naive_row['minority_f1']:.5f}, BenFPR={naive_row['benign_fpr']:.5f}, "
            f"Overrides={naive_row['overrides_total']}"
        )
        print(
            f"  [{tax}] Val-Gated Best: tau_r={best_gated['tau_rare']}, tau_c={best_gated['tau_cls']}, "
            f"tau_e4={best_gated['tau_e4_keep']} | Acc={best_gated['accuracy']:.5f}, MacroF1={best_gated['macro_f1']:.5f}, "
            f"MinF1={best_gated['minority_f1']:.5f}, BenFPR={best_gated['benign_fpr']:.5f}, "
            f"Overrides={best_gated['overrides_total']} (good={best_gated['overrides_beneficial_wrong_to_right']}, "
            f"bad={best_gated['overrides_harmful_right_to_wrong']})"
        )

    with open(CONFIGS_DIR / "exp3_selected_policies.json", "w") as f:
        json.dump(selected_policies, f, indent=2)

    print(f"\n[4/4] Experiment 3 completed in {time.perf_counter() - t0_all:.1f}s.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(force=args.force)
