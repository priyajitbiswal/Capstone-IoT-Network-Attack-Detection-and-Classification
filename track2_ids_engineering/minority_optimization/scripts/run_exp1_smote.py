"""
Experiment 1 Runner: Controlled Train-Only SMOTE + Feature Realism Audit.

Strict isolation guarantees:
  - Oversampling applied EXCLUSIVELY on training rows (350,821).
  - Unweighted LightGBM (`E4_LGBM_PARAMS`, alpha=0.0) so SMOTE is isolated from
    Experiment 2 class weighting.
  - Validation set (75,172) untouched and used solely for evaluation/selection.
  - Internal Test and Cross-Capture are NEVER loaded or inspected here.
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

from track1_dataset_evaluation.src.config import RANDOM_SEED, THIRTY_FOUR_CLASSES
from track2_ids_engineering.hierarchical_ids.src.data_utils import (
    FLOOD_CLASSES,
    MINORITY_ATTACK_CLASSES,
    NON_FLOOD_CLASSES,
    load_strategy1_preprocessor,
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
    evaluate_predictions_full,
    load_all_partitions_read_only,
)
from track2_ids_engineering.minority_optimization.src.smote_audit import (
    audit_smote_realism,
    generate_smote_train_dataset,
)

SMOTE_MODES: List[str] = [
    "SMOTE_Std_Cap4000",
    "SMOTE_ProtoSafe_Cap4000",
    "SMOTE_ProtoSafe_Cap8000",
    "SMOTE_ProtoSafe_Selective",
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
        "mode": str(best["smote_mode"]),
        "passed_guardrail": bool(guard.any()),
        "val_accuracy": float(best["accuracy"]),
        "val_macro_f1": float(best["macro_f1"]),
        "val_minority_f1": float(best["minority_f1"]),
        "val_minority_recall": float(best["minority_recall"]),
        "val_benign_fpr": float(best["benign_fpr"]),
    }


def main(force: bool = False, taxonomy_filter: str = "all") -> None:
    t0_all = time.perf_counter()
    print("=== EXPERIMENT 1: TRAIN-ONLY SMOTE & REALISM AUDIT ===")
    prep = load_strategy1_preprocessor()
    splits = load_all_partitions_read_only(include_test=False, include_cross_capture=False)
    X_train = splits["train"]["X"]
    y_train_34 = splits["train"]["y_fine"]
    y_train_29 = splits["train"]["y_op29"]
    X_val = splits["val"]["X"]

    exp_model_dir = MODELS_DIR / "exp1_smote"
    exp_model_dir.mkdir(parents=True, exist_ok=True)

    # 1. Realism Audit on Train only (cached unless --force)
    audit_csv = RESULTS_DIR / "exp1_smote_realism_audit.csv"
    if audit_csv.exists() and not force:
        print(f"\n[1/3] Loading cached realism audit -> {audit_csv.name}")
        df_audit = pd.read_csv(audit_csv)
    else:
        print("\n[1/3] Auditing SMOTE physical realism & 5-NN overlap on Training set (350,821 rows)...")
        t_audit = time.perf_counter()
        df_audit = audit_smote_realism(X_train, y_train_34, prep, random_state=RANDOM_SEED)
        df_audit.to_csv(audit_csv, index=False)
        print(f"Realism audit completed in {time.perf_counter() - t_audit:.1f}s -> {audit_csv.name}")
    print(
        df_audit[
            [
                "class_34",
                "train_count",
                "real_5nn_same_class_mean",
                "real_pct_seeds_ge1_same_5nn",
                "std_proto_pair_mismatch_pct",
                "std_variance_identity_err_pct",
                "std_syn_1nn_wrong_class_pct",
                "protosafe_syn_1nn_wrong_class_pct",
            ]
        ].to_string(index=False)
    )

    # Read baseline val accuracy from Phase 1 for deterministic guardrail logging
    df_base = pd.read_csv(RESULTS_DIR / "phase1_baseline_e4.csv")
    base_acc_34 = float(df_base[(df_base["model"] == "E4_Baseline_34") & (df_base["split"] == "val")]["accuracy"].iloc[0])
    base_acc_29 = float(df_base[(df_base["model"] == "E4_Baseline_OP29_Remap") & (df_base["split"] == "val")]["accuracy"].iloc[0])

    taxonomy_specs = [
        {
            "key": "ORIG34",
            "short": "34",
            "y_train": y_train_34,
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
            "y_train": y_train_29,
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

    sel_path = CONFIGS_DIR / "exp1_selected_smote.json"
    if sel_path.exists():
        with open(sel_path, "r") as f:
            prev_cfg = json.load(f)
        selections: Dict[str, Any] = prev_cfg.get("selections", {})
        generation_logs: Dict[str, Any] = prev_cfg.get("generation_logs", {})
    else:
        selections = {}
        generation_logs = {}

    for spec in taxonomy_specs:
        tax = spec["key"]
        print(f"\n[2/3] Training & Validation sweep for taxonomy={tax} ({len(spec['class_names'])} classes)...")
        val_rows: List[Dict[str, Any]] = []
        per_class_val_rows: List[Dict[str, Any]] = []

        for mode in SMOTE_MODES:
            model_name = f"Exp1_{mode}_{spec['short']}"
            ckpt_path = exp_model_dir / f"{model_name}.joblib"
            meta_path = exp_model_dir / f"{model_name}_meta.json"

            if ckpt_path.exists() and meta_path.exists() and not force:
                print(f"  - Loading cached checkpoint: {ckpt_path.name}")
                clf = joblib.load(ckpt_path)
                with open(meta_path, "r") as f:
                    gen_meta = json.load(f)
            else:
                t_m = time.perf_counter()
                X_aug, y_aug, gen_meta = generate_smote_train_dataset(
                    X_train_scaled=X_train,
                    y_train_34=y_train_34,
                    y_train_target=spec["y_train"],
                    mode=mode,
                    prep=prep,
                    target_names=spec["class_names"],
                    random_state=RANDOM_SEED,
                )
                print(
                    f"  - Fitting {model_name} (Train rows: {gen_meta['train_rows_before']:,} -> "
                    f"{gen_meta['train_rows_after']:,}, +{gen_meta['synthetic_rows_added']:,} syn)..."
                )
                clf = lgb.LGBMClassifier(**E4_LGBM_PARAMS)
                clf.fit(X_aug, y_aug)
                gen_meta["train_time_sec"] = round(time.perf_counter() - t_m, 2)
                joblib.dump(clf, ckpt_path, compress=3)
                with open(meta_path, "w") as f:
                    json.dump(gen_meta, f, indent=2)
                print(f"    Saved {ckpt_path.name} ({gen_meta['train_time_sec']}s)")

            generation_logs[model_name] = gen_meta
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
                    "smote_mode": mode,
                    "synthetic_rows_added": gen_meta["synthetic_rows_added"],
                    "train_time_sec": gen_meta.get("train_time_sec"),
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
        df_val.to_csv(RESULTS_DIR / f"exp1_smote_val_sweep_{spec['short']}.csv", index=False)
        pd.DataFrame(per_class_val_rows).to_csv(
            RESULTS_DIR / f"exp1_smote_val_per_class_{spec['short']}.csv", index=False
        )
        selections[tax] = _select_val_winner(df_val, spec["base_val_acc"])
        print(f"  -> Val-selected SMOTE ({tax}): {selections[tax]}")

    out_cfg = {
        "selections": selections,
        "generation_logs": generation_logs,
    }
    with open(CONFIGS_DIR / "exp1_selected_smote.json", "w") as f:
        json.dump(out_cfg, f, indent=2)

    print(f"\n[3/3] Experiment 1 completed in {time.perf_counter() - t0_all:.1f}s.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--taxonomy", choices=["all", "ORIG34", "OP29"], default="all")
    args = parser.parse_args()
    main(force=args.force, taxonomy_filter=args.taxonomy)
