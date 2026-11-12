"""
Phase 4 Read-Only Final Evaluation Runner (ORIG34 & OP29).

Loads frozen Validation-selected checkpoints & policies from Phase 1 + Experiments 1-3,
evaluates across Validation (75,172), Internal Test (75,105), and Cross-Capture (57,280),
measures latency (P50/P95 single-flow & batch throughput) and model size, and saves:
  - results/phase4_summary_34.csv, phase4_summary_op29.csv
  - results/phase4_per_class_test_34.csv, phase4_per_class_cross_34.csv
  - results/phase4_per_class_test_op29.csv, phase4_per_class_cross_op29.csv
  - results/phase4_exp3_routing_diagnostics.csv
  - results/confusion_matrices/*.npy
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd

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
    CM_DIR,
    CONFIGS_DIR,
    MODELS_DIR,
    OP29_CLASSES,
    OP29_FLOOD_CLASSES,
    OP29_MINORITY_CLASSES,
    OP29_NON_FLOOD_CLASSES,
    PROJ_34_TO_29,
    RESULTS_DIR,
    OP29RemappedE4Adapter,
    SklearnModelAdapter,
    evaluate_predictions_full,
    load_all_partitions_read_only,
)
from track2_ids_engineering.minority_optimization.src.specialist import (
    HybridE4SpecialistAdapter,
    analyze_routing_and_override_outcomes,
)


def main() -> None:
    t0_all = time.perf_counter()
    print("=== PHASE 4: READ-ONLY EVALUATION ON VAL, INTERNAL TEST & CROSS-CAPTURE ===")

    with open(CONFIGS_DIR / "exp1_selected_smote.json", "r") as f:
        exp1_cfg = json.load(f)
    with open(CONFIGS_DIR / "exp2_selected_weights.json", "r") as f:
        exp2_cfg = json.load(f)
    with open(CONFIGS_DIR / "exp3_selected_policies.json", "r") as f:
        exp3_cfg = json.load(f)

    splits = load_all_partitions_read_only(include_test=True, include_cross_capture=True)

    e4_ckpt = HIER_MODELS_DIR / "e4_lgbm_leaves63_sqrt.joblib"
    e4_model = joblib.load(e4_ckpt)
    stage_a_ckpt = Path(exp3_cfg["stage_a_checkpoint"])
    stage_b_ckpt = Path(exp3_cfg["stage_b_checkpoint"])
    stage_a_model = joblib.load(stage_a_ckpt)
    stage_b_model = joblib.load(stage_b_ckpt)

    smote_modes = [
        "SMOTE_Std_Cap4000",
        "SMOTE_ProtoSafe_Cap4000",
        "SMOTE_ProtoSafe_Cap8000",
        "SMOTE_ProtoSafe_Selective",
    ]
    weight_tags = [
        ("Power_a025", 0.25),
        ("SqrtBalanced_a050", 0.50),
        ("Power_a075", 0.75),
        ("Balanced_a100", 1.00),
    ]

    routing_rows: List[Dict[str, Any]] = []

    for tax, short, y_key, cls_names, min_cls, fl_cls, nf_cls in [
        ("ORIG34", "34", "y_fine", THIRTY_FOUR_CLASSES, MINORITY_ATTACK_CLASSES, FLOOD_CLASSES, NON_FLOOD_CLASSES),
        ("OP29", "op29", "y_op29", OP29_CLASSES, OP29_MINORITY_CLASSES, OP29_FLOOD_CLASSES, OP29_NON_FLOOD_CLASSES),
    ]:
        print(f"\nEvaluating all frozen models for taxonomy={tax} ({len(cls_names)} classes)...")
        best_smote_name = exp1_cfg["selections"][tax]["selected_model"]
        best_wt_name = exp2_cfg["selections"][tax]["selected_model"]
        pol_naive = exp3_cfg["policies"][tax]["naive"]
        pol_gated = exp3_cfg["policies"][tax]["gated"]

        # Assemble candidate roster deterministically
        roster: List[Dict[str, Any]] = []
        if tax == "ORIG34":
            roster.append(
                {
                    "group": "Baseline",
                    "is_val_selected": True,
                    "adapter": SklearnModelAdapter("E4_Baseline_34", e4_model, e4_ckpt),
                    "save_cm": True,
                }
            )
        else:
            roster.append(
                {
                    "group": "Baseline",
                    "is_val_selected": True,
                    "adapter": OP29RemappedE4Adapter(e4_model, e4_ckpt, "E4_Baseline_OP29_Remap"),
                    "save_cm": True,
                }
            )

        for sm in smote_modes:
            mname = f"Exp1_{sm}_{short}"
            ckpt = MODELS_DIR / "exp1_smote" / f"{mname}.joblib"
            clf = joblib.load(ckpt)
            roster.append(
                {
                    "group": "Exp1_SMOTE",
                    "is_val_selected": (mname == best_smote_name),
                    "adapter": SklearnModelAdapter(mname, clf, ckpt),
                    "save_cm": (mname == best_smote_name or sm == "SMOTE_Std_Cap4000"),
                }
            )

        for w_label, alpha in weight_tags:
            mname = f"Exp2_{w_label}_{short}"
            ckpt = MODELS_DIR / "exp2_weights" / f"{mname}.joblib"
            clf = joblib.load(ckpt)
            roster.append(
                {
                    "group": f"Exp2_Weight(a={alpha:.2f})",
                    "is_val_selected": (mname == best_wt_name),
                    "adapter": SklearnModelAdapter(mname, clf, ckpt),
                    "save_cm": True,
                }
            )

        for ptag, pcfg, is_sel in [
            ("Exp3_Hybrid_Naive", pol_naive, False),
            ("Exp3_Hybrid_Gated", pol_gated, True),
        ]:
            mname = f"{ptag}_{short}"
            hyb = HybridE4SpecialistAdapter(
                name=mname,
                e4_model=e4_model,
                stage_a_model=stage_a_model,
                stage_b_model=stage_b_model,
                taxonomy=tax,
                tau_rare=pcfg["tau_rare"],
                tau_cls=pcfg["tau_cls"],
                tau_e4_keep=pcfg["tau_e4_keep"],
                checkpoint_paths=[e4_ckpt, stage_a_ckpt, stage_b_ckpt],
            )
            roster.append(
                {
                    "group": f"Exp3_Specialist({ptag})",
                    "is_val_selected": is_sel,
                    "adapter": hyb,
                    "save_cm": True,
                    "hyb_policy": pcfg,
                }
            )

        summary_rows: List[Dict[str, Any]] = []
        per_class_by_split: Dict[str, List[Dict[str, Any]]] = {"test": [], "cross": []}

        # Precompute E4 base predictions for routing decomposition of Exp3 models
        e4_preds = {}
        for sp_name in ["val", "test", "cross"]:
            p34 = e4_model.predict_proba(splits[sp_name]["X"])
            p_tax = p34 if tax == "ORIG34" else (p34 @ PROJ_34_TO_29)
            e4_preds[sp_name] = np.argmax(p_tax, axis=1)

        for entry in roster:
            adapter = entry["adapter"]
            mname = adapter.name
            for sp_name in ["val", "test", "cross"]:
                sp = splits[sp_name]
                X = sp["X"]
                y_true = sp[y_key]
                if isinstance(adapter, HybridE4SpecialistAdapter):
                    y_pred, y_prob, hyb_diag = adapter._forward(X)
                    rout = analyze_routing_and_override_outcomes(y_true, e4_preds[sp_name], y_pred, tax)
                    routing_rows.append(
                        {
                            "model": mname,
                            "taxonomy": tax,
                            "split": sp_name,
                            "tau_rare": adapter.tau_rare,
                            "tau_cls": adapter.tau_cls,
                            "tau_e4_keep": adapter.tau_e4_keep,
                            **hyb_diag,
                            **rout,
                        }
                    )
                else:
                    y_prob = adapter.predict_proba(X)
                    y_pred = np.argmax(y_prob, axis=1)

                res = evaluate_predictions_full(
                    model_name=mname,
                    split_name=sp_name,
                    taxonomy=tax,
                    y_true=y_true,
                    y_pred=y_pred,
                    y_prob=y_prob,
                    class_names=cls_names,
                    minority_classes=min_cls,
                    flood_classes=fl_cls,
                    non_flood_classes=nf_cls,
                    model_obj_for_latency=adapter if sp_name == "test" else None,
                    X_for_latency=X if sp_name == "test" else None,
                    model_size_bytes=adapter.model_size_bytes(),
                    extra_meta={
                        "experiment_group": entry["group"],
                        "is_val_selected": entry["is_val_selected"],
                    },
                )
                summary_rows.append(res["summary"])
                if sp_name in ("test", "cross"):
                    per_class_by_split[sp_name].extend(res["per_class_rows"])
                if entry["save_cm"] and sp_name in ("test", "cross"):
                    np.save(CM_DIR / f"{mname}_{sp_name}_{short}.npy", res["confusion_matrix"])

            s_test = [r for r in summary_rows if r["model"] == mname and r["split"] == "test"][0]
            s_cross = [r for r in summary_rows if r["model"] == mname and r["split"] == "cross"][0]
            print(
                f"  {mname:34s} | Test Acc={s_test['accuracy']:.5f} MF1={s_test['macro_f1']:.5f} "
                f"MinF1={s_test['minority_f1']:.5f} BenFPR={s_test['benign_fpr']:.5f} "
                f"P50={s_test.get('p50_latency_ms', 0):.3f}ms | Cross MF1={s_cross['macro_f1']:.5f} "
                f"MinF1={s_cross['minority_f1']:.5f}"
            )

        pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / f"phase4_summary_{short}.csv", index=False)
        pd.DataFrame(per_class_by_split["test"]).to_csv(
            RESULTS_DIR / f"phase4_per_class_test_{short}.csv", index=False
        )
        pd.DataFrame(per_class_by_split["cross"]).to_csv(
            RESULTS_DIR / f"phase4_per_class_cross_{short}.csv", index=False
        )

    pd.DataFrame(routing_rows).to_csv(RESULTS_DIR / "phase4_exp3_routing_diagnostics.csv", index=False)
    print(f"\nPhase 4 evaluation completed in {time.perf_counter() - t0_all:.1f}s.")


if __name__ == "__main__":
    main()
