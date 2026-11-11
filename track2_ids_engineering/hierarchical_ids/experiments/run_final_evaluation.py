"""
Phases 5-6: Strictly Read-Only Final Evaluation & Error Analysis for E1-E4.

Loads frozen configurations selected strictly on Validation (`frozen_architectures_config.json`)
and evaluates all architectures across:
  1. Validation (N = 75,172)
  2. Internal Test (N = 75,105)
  3. Cross-Capture Clean Holdout (N = 57,280)

Generates master comparison CSVs, per-class metrics, minority attack tables,
34x34 confusion matrices, E2 stage decomposition, E3 fusion disagreement analysis,
and operational single-flow latency / throughput benchmarks.
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

import track1_dataset_evaluation.src as _t1_compat  # noqa: F401
from track1_dataset_evaluation.src.config import (
    MODELS_V2_DIR as TRACK1_MODELS_DIR,
    RANDOM_SEED,
    THIRTY_FOUR_CLASSES,
)
from track2_ids_engineering.hierarchical_ids.src.architectures import (
    BaseIDSArchitecture,
    Flat34BaselineArchitecture,
    HierarchicalCascade3Stage,
    ParallelSpecialistFusion,
)
from track2_ids_engineering.hierarchical_ids.src.data_utils import (
    HIER_RESULTS_DIR,
    MINORITY_ATTACK_CLASSES,
    load_track1_splits,
)
from track2_ids_engineering.hierarchical_ids.src.metrics_eval import evaluate_end_to_end


def build_frozen_architectures(cfg: Dict[str, Any]) -> List[BaseIDSArchitecture]:
    """Instantiates all frozen E1-E4 architectures purely from saved checkpoints."""
    archs: List[BaseIDSArchitecture] = []

    # Cache loaded joblib objects by path to save RAM
    loaded_cache: Dict[str, Any] = {}

    def _load(p_str: str) -> Any:
        if p_str not in loaded_cache:
            loaded_cache[p_str] = joblib.load(p_str)
        return loaded_cache[p_str]

    # 1. E1 Flat 34-class baseline (Track 1)
    c_e1 = cfg["E1"]
    archs.append(
        Flat34BaselineArchitecture(
            name=c_e1["name"],
            model=_load(c_e1["checkpoint"]),
            checkpoint_path=Path(c_e1["checkpoint"]),
        )
    )

    # 2. E2 Default (T=0.50) & E2 Validation-Tuned
    for key in ["E2_default", "E2_tuned"]:
        ce = cfg[key]
        archs.append(
            HierarchicalCascade3Stage(
                name=ce["name"],
                stage1_model=_load(ce["s1_ckpt"]),
                stage2_model=_load(ce["s2_ckpt"]),
                flood_model=_load(ce["stage3a_ckpt"]),
                non_flood_model=_load(ce["stage3b_ckpt"]),
                stage2_mode=ce["s2_mode"],
                attack_threshold=ce["attack_threshold"],
                flood_route_threshold=ce["flood_route_threshold"],
                checkpoint_paths=[
                    Path(ce["s1_ckpt"]),
                    Path(ce["s2_ckpt"]),
                    Path(ce["stage3a_ckpt"]),
                    Path(ce["stage3b_ckpt"]),
                ],
            )
        )

    # 3. E3 Variants: NaiveMax, TempAlpha, MetaGate
    c_e3n = cfg["E3_naive"]
    archs.append(
        ParallelSpecialistFusion(
            name=c_e3n["name"],
            stage1_model=_load(c_e3n["s1_ckpt"]),
            flood_model=_load(c_e3n["stage3a_ckpt"]),
            non_flood_model=_load(c_e3n["stage3b_ckpt"]),
            fusion_mode="naive_max",
            attack_threshold=c_e3n["attack_threshold"],
            checkpoint_paths=[
                Path(c_e3n["s1_ckpt"]),
                Path(c_e3n["stage3a_ckpt"]),
                Path(c_e3n["stage3b_ckpt"]),
            ],
        )
    )

    c_e3a = cfg["E3_temp_alpha"]
    archs.append(
        ParallelSpecialistFusion(
            name=c_e3a["name"],
            stage1_model=_load(c_e3a["s1_ckpt"]),
            flood_model=_load(c_e3a["stage3a_ckpt"]),
            non_flood_model=_load(c_e3a["stage3b_ckpt"]),
            fusion_mode="temp_alpha",
            attack_threshold=c_e3a["attack_threshold"],
            temp_flood=c_e3a["temp_flood"],
            temp_non_flood=c_e3a["temp_non_flood"],
            alpha_flood=c_e3a["alpha_flood"],
            checkpoint_paths=[
                Path(c_e3a["s1_ckpt"]),
                Path(c_e3a["stage3a_ckpt"]),
                Path(c_e3a["stage3b_ckpt"]),
            ],
        )
    )

    c_e3m = cfg["E3_meta_gate"]
    archs.append(
        ParallelSpecialistFusion(
            name=c_e3m["name"],
            stage1_model=_load(c_e3m["s1_ckpt"]),
            flood_model=_load(c_e3m["stage3a_ckpt"]),
            non_flood_model=_load(c_e3m["stage3b_ckpt"]),
            fusion_mode="meta_gate_lr",
            attack_threshold=c_e3m["attack_threshold"],
            temp_flood=c_e3m["temp_flood"],
            temp_non_flood=c_e3m["temp_non_flood"],
            meta_gate_model=_load(c_e3m["meta_gate_ckpt"]),
            checkpoint_paths=[
                Path(c_e3m["s1_ckpt"]),
                Path(c_e3m["stage3a_ckpt"]),
                Path(c_e3m["stage3b_ckpt"]),
                Path(c_e3m["meta_gate_ckpt"]),
            ],
        )
    )

    # 4. E4 Improved Flat 34-class
    c_e4 = cfg["E4"]
    bias = np.asarray(c_e4["class_logit_bias"], dtype=np.float64) if c_e4["class_logit_bias"] is not None else None
    archs.append(
        Flat34BaselineArchitecture(
            name=f"{c_e4['name']} ({c_e4['base_candidate_id']})",
            model=_load(c_e4["checkpoint"]),
            checkpoint_path=Path(c_e4["checkpoint"]),
            temperature=c_e4["temperature"],
            class_logit_bias=bias,
        )
    )

    # Also include Track 1 Unweighted reference for completeness if E4 winner wasn't unweighted
    unw_ckpt = TRACK1_MODELS_DIR / "ablation_unweighted_fine_grained_lightgbm.joblib"
    if str(unw_ckpt) != c_e4["checkpoint"] and unw_ckpt.exists():
        archs.append(
            Flat34BaselineArchitecture(
                name="E1_Ref_Unweighted_Track1",
                model=_load(str(unw_ckpt)),
                checkpoint_path=unw_ckpt,
            )
        )

    return archs


def main() -> None:
    t0 = time.perf_counter()
    cfg_path = HIER_RESULTS_DIR / "frozen_architectures_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing frozen config: {cfg_path}. Run train_and_tune_val.py first.")

    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    print("=" * 80)
    print("PHASES 5-6: READ-ONLY EVALUATION ON VAL, INTERNAL TEST & CROSS-CAPTURE")
    print("=" * 80)

    splits = load_track1_splits(include_test=True, include_cross_capture=True, verify_disjoint=True)
    archs = build_frozen_architectures(cfg)

    cm_dir = HIER_RESULTS_DIR / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(RANDOM_SEED)

    for split_name in ["val", "test", "cross"]:
        sp = splits[split_name]
        X_raw = sp["X"]
        # Deterministic shuffled view for representative single-flow latency subset
        perm = rng.permutation(len(X_raw))
        X_lat = X_raw[perm]
        run_lat = split_name in ("test", "cross")

        print(f"\n--- Evaluating Split: {split_name.upper()} (N={len(X_raw):,}) ---", flush=True)
        summary_rows: List[Dict[str, Any]] = []
        per_class_rows: List[Dict[str, Any]] = []
        minority_rows: List[Dict[str, Any]] = []
        err_rows: List[Dict[str, Any]] = []
        stage_diag_map: Dict[str, Any] = {}
        trace_store: Dict[str, Dict[str, Any]] = {}

        for arch in archs:
            # Adjust single-flow sample count if RF is Stage 1 (expensive single-flow tree traversal)
            ckpt_list = getattr(arch, "checkpoint_paths", [getattr(arch, "checkpoint_path", None)])
            has_rf = any("random_forest" in str(p).lower() for p in ckpt_list if p is not None)
            n_single = 200 if has_rf else 1000
            n_runs = 1 if has_rf else 3

            # Run end-to-end predictions on full split (ordered)
            res = evaluate_end_to_end(
                arch=arch,
                X=X_raw,
                y_fine_true=sp["y_fine"],
                y_cat_true=sp["y_category"],
                is_attack_true=sp["is_attack"],
                is_flood_true=sp["is_flood"],
                is_non_flood_true=sp["is_non_flood"],
                run_latency_bench=False,
            )

            if run_lat:
                # Benchmark latency on shuffled subset so Benign/Flood/Non-Flood appear in natural proportion
                from track1_dataset_evaluation.src.evaluate import (
                    benchmark_batch_throughput,
                    benchmark_single_flow_latency,
                )
                sb = benchmark_single_flow_latency(
                    model=arch,
                    X_sample=X_lat,
                    n_samples=n_single,
                    n_warmup=min(100, n_single // 4),
                    n_runs=n_runs,
                    scaler=None,
                )
                bb = benchmark_batch_throughput(
                    model=arch,
                    X_sample=X_lat,
                    batch_size=min(20000, len(X_lat)),
                    n_warmup=2,
                    n_runs=5,
                    scaler=None,
                )
                res["summary"]["single_latency_p50_us"] = sb["median_latency_us"]
                res["summary"]["single_latency_p95_us"] = sb["p95_latency_us"]
                res["summary"]["single_latency_p99_us"] = sb["p99_latency_us"]
                res["summary"]["batch_throughput_fps"] = bb["throughput_predictions_per_sec"]
                res["summary"]["batch_amortized_us"] = bb["amortized_latency_us"]

            s = res["summary"]
            ed = s.pop("error_decomposition")
            mp = s.pop("minority_per_class")
            summary_rows.append(s)
            err_rows.append({"architecture": arch.name, "n_samples": s["n_samples"], **ed})

            # Per-class long-format rows
            rep = res["per_class_report"]
            for cls_name in THIRTY_FOUR_CLASSES:
                per_class_rows.append({
                    "architecture": arch.name,
                    "class": cls_name,
                    "precision": round(float(rep[cls_name]["precision"]), 5),
                    "recall": round(float(rep[cls_name]["recall"]), 5),
                    "f1": round(float(rep[cls_name]["f1-score"]), 5),
                    "support": int(rep[cls_name]["support"]),
                })

            # Minority wide row
            min_row = {
                "architecture": arch.name,
                "minority_macro_recall": s["minority_macro_recall"],
                "minority_macro_f1": s["minority_macro_f1"],
            }
            for mc in MINORITY_ATTACK_CLASSES:
                min_row[f"{mc}_recall"] = mp[mc]["recall"]
                min_row[f"{mc}_f1"] = mp[mc]["f1"]
            minority_rows.append(min_row)

            # Save confusion matrix
            safe_name = arch.name.replace(" ", "_").replace("(", "").replace(")", "")
            np.save(cm_dir / f"cm_{safe_name}_{split_name}.npy", res["confusion_matrix"])

            if res["stage_diagnostics"] is not None:
                stage_diag_map[arch.name] = res["stage_diagnostics"]

            # Store predictions for E2 vs E3 head-to-head error analysis on test & cross
            if split_name in ("test", "cross"):
                trace_store[arch.name] = arch.predict_with_trace(X_raw)

            lat_str = f" | P50={s.get('single_latency_p50_us', 0):.1f}us" if run_lat else ""
            print(
                f"  {arch.name:<45} | Acc={s['accuracy']:.4f} | MacroF1={s['macro_f1']:.4f} | "
                f"MCC={s['mcc']:.4f} | MinRec={s['minority_macro_recall']:.4f} | "
                f"FloodF1={s['flood_macro_f1']:.4f}{lat_str}",
                flush=True,
            )

        pd.DataFrame(summary_rows).to_csv(HIER_RESULTS_DIR / f"final_comparison_{split_name}.csv", index=False)
        pd.DataFrame(per_class_rows).to_csv(HIER_RESULTS_DIR / f"per_class_metrics_{split_name}.csv", index=False)
        pd.DataFrame(minority_rows).to_csv(HIER_RESULTS_DIR / f"minority_attack_comparison_{split_name}.csv", index=False)
        pd.DataFrame(err_rows).to_csv(HIER_RESULTS_DIR / f"error_attribution_{split_name}.csv", index=False)
        with open(HIER_RESULTS_DIR / f"stage_diagnostics_{split_name}.json", "w") as f:
            json.dump(stage_diag_map, f, indent=2)

        # Pairwise E1 vs E2 vs E3 vs E4 Error-Recovery & Routing Analysis on Test & Cross
        if split_name in ("test", "cross"):
            y_true = sp["y_fine"]
            e1_pred = trace_store[cfg["E1"]["name"]]["y_pred"]
            e2_pred = trace_store[cfg["E2_tuned"]["name"]]["y_pred"]
            e3_pred = trace_store[cfg["E3_meta_gate"]["name"]]["y_pred"]
            e4_name = f"{cfg['E4']['name']} ({cfg['E4']['base_candidate_id']})"
            e4_pred = trace_store[e4_name]["y_pred"]

            e2_tr = trace_store[cfg["E2_tuned"]["name"]]
            e2_route_flood = e2_tr["specialist_route"] == "3A_FLOOD"
            e2_route_nf = e2_tr["specialist_route"] == "3B_NON_FLOOD"

            # Where did E2 fail solely due to Stage 2 misrouting?
            e2_misrouted_flood = sp["is_flood"] & e2_route_nf
            e2_misrouted_nf = sp["is_non_flood"] & e2_route_flood
            e2_misrouted_any = e2_misrouted_flood | e2_misrouted_nf

            pairwise = {
                "split": split_name,
                "n_samples": int(len(y_true)),
                "E1_vs_E2_tuned": {
                    "both_correct": int(np.sum((e1_pred == y_true) & (e2_pred == y_true))),
                    "E2_fixed_E1_error": int(np.sum((e1_pred != y_true) & (e2_pred == y_true))),
                    "E2_broke_E1_correct": int(np.sum((e1_pred == y_true) & (e2_pred != y_true))),
                    "both_wrong": int(np.sum((e1_pred != y_true) & (e2_pred != y_true))),
                },
                "E1_vs_E3_meta_gate": {
                    "both_correct": int(np.sum((e1_pred == y_true) & (e3_pred == y_true))),
                    "E3_fixed_E1_error": int(np.sum((e1_pred != y_true) & (e3_pred == y_true))),
                    "E3_broke_E1_correct": int(np.sum((e1_pred == y_true) & (e3_pred != y_true))),
                    "both_wrong": int(np.sum((e1_pred != y_true) & (e3_pred != y_true))),
                },
                "E2_vs_E3_meta_gate": {
                    "E2_stage2_misrouted_total": int(np.sum(e2_misrouted_any)),
                    "E2_misrouted_recovered_by_E3": int(np.sum(e2_misrouted_any & (e3_pred == y_true))),
                    "E3_fixed_E2_error": int(np.sum((e2_pred != y_true) & (e3_pred == y_true))),
                    "E3_broke_E2_correct": int(np.sum((e2_pred == y_true) & (e3_pred != y_true))),
                },
                "E1_vs_E4": {
                    "both_correct": int(np.sum((e1_pred == y_true) & (e4_pred == y_true))),
                    "E4_fixed_E1_error": int(np.sum((e1_pred != y_true) & (e4_pred == y_true))),
                    "E4_broke_E1_correct": int(np.sum((e1_pred == y_true) & (e4_pred != y_true))),
                },
            }
            with open(HIER_RESULTS_DIR / f"pairwise_error_analysis_{split_name}.json", "w") as f:
                json.dump(pairwise, f, indent=2)

    elapsed = time.perf_counter() - t0
    print("\n" + "=" * 80)
    print(f"PHASES 5-6 COMPLETE IN {elapsed:.1f}s")
    print(f"All evaluation tables & JSONs saved under: {HIER_RESULTS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
