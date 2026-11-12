"""
Experiment 2 Runner: Isolated Class-Weight Power Optimization (No SMOTE).

Trains LightGBM models (`E4_LGBM_PARAMS`) exclusively on the original Training set
(350,821 rows, SMOTE disabled) with normalized inverse-frequency power weights:
    w_c proportional to (N_c)^(-alpha), scaled so mean_c(w_c) = 1.0
Tested alphas:
  - 0.25 -> Power_a025
  - 0.50 -> SqrtBalanced_a050
  - 0.75 -> Power_a075
  - 1.00 -> Balanced_a100 (standard sklearn 'balanced' up to global scalar)

Runs independently for 34-class (`ORIG34`) and merged-flood 29-class (`OP29`).
Validation set (75,172 rows) is used strictly for selection; Test & Cross-Capture
are never touched here.
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

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from track1_dataset_evaluation.src.config import THIRTY_FOUR_CLASSES
from track2_ids_engineering.hierarchical_ids.src.data_utils import (
    FLOOD_CLASSES,
    MINORITY_ATTACK_CLASSES,
    NON_FLOOD_CLASSES,
)
from track2_ids_engineering.minority_optimization.src.common import (
    CONFIGS_DIR,
    E4_LGBM_PARAMS,
    MODELS_DIR,
    OP29_CLASSES,
    OP29_FLOOD_CLASSES,
    OP29_MINORITY_CLASSES,
    OP29_NON_FLOOD_CLASSES,
    RESULTS_DIR,
    SklearnModelAdapter,
    compute_power_sample_weights,
    evaluate_predictions_full,
    load_all_partitions_read_only,
)

WEIGHT_SCHEDULE = [
    (0.25, "Power_a025"),
    (0.50, "SqrtBalanced_a050"),
    (0.75, "Power_a075"),
    (1.00, "Balanced_a100"),
]


def _select_val_winner(df_val: pd.DataFrame, baseline_acc: float) -> Dict[str, Any]:
    """Deterministic Validation-only selection balancing Macro-F1 and Minority-F1."""
    df = df_val.copy()
    df["composite_score"] = 0.5 * df["macro_f1"] + 0.5 * df["minority_f1"]
    guard = (df["benign_fpr"] <= 0.25) & (df["accuracy"] >= baseline_acc - 0.03)
    pool = df[guard] if guard.any() else df
    best = pool.sort_values(
        by=["composite_score", "minority_f1", "macro_f1", "accuracy"],
        ascending=False,
    ).iloc[0]
    return {
        "selected_model": str(best["model"]),
        "weight_label": str(best["weight_label"]),
        "alpha": float(best["weight_alpha"]),
        "passed_guardrail": bool(guard.any()),
        "val_accuracy": float(best["accuracy"]),
        "val_macro_f1": float(best["macro_f1"]),
        "val_minority_f1": float(best["minority_f1"]),
        "val_minority_recall": float(best["minority_recall"]),
        "val_benign_fpr": float(best["benign_fpr"]),
    }


def main(force: bool = False, taxonomy_filter: str = "all") -> None:
    t0_all = time.perf_counter()
    print("=== EXPERIMENT 2: ISOLATED CLASS-WEIGHT POWER OPTIMIZATION (NO SMOTE) ===")
    splits = load_all_partitions_read_only(include_test=False, include_cross_capture=False)
    X_train = splits["train"]["X"]
    X_val = splits["val"]["X"]

    exp_model_dir = MODELS_DIR / "exp2_weights"
    exp_model_dir.mkdir(parents=True, exist_ok=True)

    df_base = pd.read_csv(RESULTS_DIR / "phase1_baseline_e4.csv")
    base_acc_34 = float(df_base[(df_base["model"] == "E4_Baseline_34") & (df_base["split"] == "val")]["accuracy"].iloc[0])
    base_acc_29 = float(df_base[(df_base["model"] == "E4_Baseline_OP29_Remap") & (df_base["split"] == "val")]["accuracy"].iloc[0])

    taxonomy_specs = [
        {
            "key": "ORIG34",
            "short": "34",
            "y_train": splits["train"]["y_fine"],
            "y_val": splits["val"]["y_fine"],
            "class_names": THIRTY_FOUR_CLASSES,
            "minority_classes": MINORITY_ATTACK_CLASSES,
            "flood_classes": FLOOD_CLASSES,
            "non_flood_classes": NON_FLOOD_CLASSES,
            "base_val_acc": base_acc_34,
        },
        {
            "key": "OP29",
            "short": "op29",
            "y_train": splits["train"]["y_op29"],
            "y_val": splits["val"]["y_op29"],
            "class_names": OP29_CLASSES,
            "minority_classes": OP29_MINORITY_CLASSES,
            "flood_classes": OP29_FLOOD_CLASSES,
            "non_flood_classes": OP29_NON_FLOOD_CLASSES,
            "base_val_acc": base_acc_29,
        },
    ]
    if taxonomy_filter != "all":
        taxonomy_specs = [s for s in taxonomy_specs if s["key"] == taxonomy_filter]

    sel_path = CONFIGS_DIR / "exp2_selected_weights.json"
    if sel_path.exists():
        with open(sel_path, "r") as f:
            selections: Dict[str, Any] = json.load(f).get("selections", {})
    else:
        selections = {}
    sched_path = RESULTS_DIR / "exp2_weight_schedules.csv"
    if sched_path.exists() and taxonomy_filter != "all":
        prev_sched = pd.read_csv(sched_path)
        prev_sched = prev_sched[prev_sched["taxonomy"] != taxonomy_filter]
        weight_schedule_rows: List[Dict[str, Any]] = prev_sched.to_dict("records")
    else:
        weight_schedule_rows = []

    for spec in taxonomy_specs:
        tax = spec["key"]
        print(f"\nTraining & Validation sweep for taxonomy={tax} ({len(spec['class_names'])} classes)...")
        y_tr = spec["y_train"]
        val_rows: List[Dict[str, Any]] = []
        per_class_val_rows: List[Dict[str, Any]] = []

        unique_ids, counts = np.unique(y_tr, return_counts=True)
        count_map = dict(zip(unique_ids.tolist(), counts.tolist()))

        for alpha, w_label in WEIGHT_SCHEDULE:
            model_name = f"Exp2_{w_label}_{spec['short']}"
            ckpt_path = exp_model_dir / f"{model_name}.joblib"
            meta_path = exp_model_dir / f"{model_name}_meta.json"

            # Record exact per-class normalized weight vector
            raw_w = {cid: (cnt ** (-alpha)) for cid, cnt in count_map.items()}
            mean_w = float(np.mean(list(raw_w.values())))
            norm_w = {cid: w / mean_w for cid, w in raw_w.items()}
            for cid, cname in enumerate(spec["class_names"]):
                weight_schedule_rows.append(
                    {
                        "taxonomy": tax,
                        "weight_label": w_label,
                        "alpha": alpha,
                        "class": cname,
                        "train_count": count_map[cid],
                        "normalized_weight": round(norm_w[cid], 6),
                        "is_minority_8": cname in spec["minority_classes"],
                    }
                )

            if ckpt_path.exists() and meta_path.exists() and not force:
                print(f"  - Loading cached checkpoint: {ckpt_path.name}")
                clf = joblib.load(ckpt_path)
                with open(meta_path, "r") as f:
                    fit_meta = json.load(f)
            else:
                t_m = time.perf_counter()
                sample_w = compute_power_sample_weights(y_tr, alpha=alpha)
                print(
                    f"  - Fitting {model_name} (alpha={alpha:.2f}, max/min class weight="
                    f"{max(norm_w.values()) / min(norm_w.values()):.2f}x)..."
                )
                clf = lgb.LGBMClassifier(**E4_LGBM_PARAMS)
                clf.fit(X_train, y_tr, sample_weight=sample_w)
                fit_meta = {
                    "model_name": model_name,
                    "taxonomy": tax,
                    "alpha": alpha,
                    "weight_label": w_label,
                    "max_min_weight_ratio": round(max(norm_w.values()) / min(norm_w.values()), 4),
                    "train_time_sec": round(time.perf_counter() - t_m, 2),
                }
                joblib.dump(clf, ckpt_path, compress=3)
                with open(meta_path, "w") as f:
                    json.dump(fit_meta, f, indent=2)
                print(f"    Saved {ckpt_path.name} ({fit_meta['train_time_sec']}s)")

            adapter = SklearnModelAdapter(model_name, clf, ckpt_path)
            p_val = adapter.predict_proba(X_val)
            yhat_val = np.argmax(p_val, axis=1)
            res = evaluate_predictions_full(
                model_name=model_name,
                split_name="val",
                taxonomy=tax,
                y_true=spec["y_val"],
                y_pred=yhat_val,
                y_prob=p_val,
                class_names=spec["class_names"],
                minority_classes=spec["minority_classes"],
                flood_classes=spec["flood_classes"],
                non_flood_classes=spec["non_flood_classes"],
                model_obj_for_latency=None,
                X_for_latency=None,
                model_size_bytes=adapter.model_size_bytes(),
                extra_meta={
                    "weight_label": w_label,
                    "weight_alpha": alpha,
                    "train_time_sec": fit_meta.get("train_time_sec"),
                },
            )
            val_rows.append(res["summary"])
            per_class_val_rows.extend(res["per_class_rows"])
            s = res["summary"]
            print(
                f"    Val {model_name}: Acc={s['accuracy']:.5f} | MacroF1={s['macro_f1']:.5f} | "
                f"MCC={s['mcc']:.5f} | MinF1={s['minority_f1']:.5f} | MinRec={s['minority_recall']:.5f} | "
                f"BenFPR={s['benign_fpr']:.5f}"
            )

        df_val = pd.DataFrame(val_rows)
        df_val.to_csv(RESULTS_DIR / f"exp2_weights_val_sweep_{spec['short']}.csv", index=False)
        pd.DataFrame(per_class_val_rows).to_csv(
            RESULTS_DIR / f"exp2_weights_val_per_class_{spec['short']}.csv", index=False
        )
        selections[tax] = _select_val_winner(df_val, spec["base_val_acc"])
        print(f"  -> Val-selected Weighting ({tax}): {selections[tax]}")

    pd.DataFrame(weight_schedule_rows).to_csv(RESULTS_DIR / "exp2_weight_schedules.csv", index=False)
    with open(CONFIGS_DIR / "exp2_selected_weights.json", "w") as f:
        json.dump({"selections": selections}, f, indent=2)

    print(f"\nExperiment 2 completed in {time.perf_counter() - t0_all:.1f}s.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--taxonomy", choices=["all", "ORIG34", "OP29"], default="all")
    args = parser.parse_args()
    main(force=args.force, taxonomy_filter=args.taxonomy)
