"""
Promoted Main Track 2 Production Pipeline: E4 + Exp3 Flood-Free Rare-Attack Specialist.

Unifies the frozen E4 Flat34 LightGBM backbone (from `hierarchical_ids/`) with the
validation-tuned Stage A (Binary Rare-vs-NonRare) + Stage B (8-Class Rare Specialist)
from `minority_optimization/` under a single top-level Track 2 API.

Why this is the promoted main pipeline:
  - Preserves E4's 99%+ Flood detection accuracy with ZERO true rare flows lost to Flood bypass.
  - Only method that improves BOTH Internal Test (Min-F1 0.2196 -> 0.2288, OP29 Macro-F1 0.6809 -> 0.6834)
    and unseen Cross-Capture generalization (Min-F1 0.2872 -> 0.2930, OP29 Acc 78.04% -> 78.16%)
    without inflating Benign false positives (~21.4% Benign FPR).
  - Supports both operational `OP29` flood-merged remediation and `ORIG34` fine attribution.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

import track1_dataset_evaluation.src as _t1_compat  # noqa: F401
from track1_dataset_evaluation.src.config import (
    FEATURE_COLUMNS,
    MODELS_V2_DIR as TRACK1_MODELS_DIR,
    THIRTY_FOUR_CLASSES,
)
from track2_ids_engineering.hierarchical_ids.src.architectures import BaseIDSArchitecture
from track2_ids_engineering.hierarchical_ids.src.data_utils import HIER_MODELS_DIR
from track2_ids_engineering.minority_optimization.src.common import (
    MIN_OPT_ROOT,
    OP29_CLASSES,
    PROJ_34_TO_29,
)
from track2_ids_engineering.minority_optimization.src.specialist import (
    FLOOD_29_IDX,
    FLOOD_34_IDX,
    RARE_8_CLASSES,
    apply_gated_hybrid_policy,
)

DEFAULT_HYBRID_POLICY_JSON = MIN_OPT_ROOT / "configs" / "exp3_selected_policies.json"
DEFAULT_E4_PATH = HIER_MODELS_DIR / "e4_lgbm_leaves63_sqrt.joblib"
DEFAULT_STAGE_A_PATH = MIN_OPT_ROOT / "models" / "exp3_specialist" / "StageA_Bin_unweighted.joblib"
DEFAULT_STAGE_B_PATH = MIN_OPT_ROOT / "models" / "exp3_specialist" / "StageB_Rare8_sqrt_a050.joblib"
DEFAULT_PREPROCESSOR_PATH = TRACK1_MODELS_DIR / "preprocessor_strategy1.joblib"

# Fallback thresholds matching configs/exp3_selected_policies.json ("gated")
DEFAULT_GATED_THRESHOLDS: Dict[str, float] = {
    "tau_rare": 0.30,
    "tau_cls": 0.25,
    "tau_e4_keep": 0.65,
}


def load_gated_policy_thresholds(
    taxonomy: Literal["OP29", "ORIG34"] = "OP29",
    policy_path: Optional[Path] = None,
) -> Dict[str, float]:
    """Loads validation-selected Exp 3 gated thresholds from config, with safe defaults."""
    cfg_path = Path(policy_path) if policy_path else DEFAULT_HYBRID_POLICY_JSON
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        gated = data.get("policies", {}).get(taxonomy, {}).get("gated", {})
        if {"tau_rare", "tau_cls", "tau_e4_keep"}.issubset(gated):
            return {
                "tau_rare": float(gated["tau_rare"]),
                "tau_cls": float(gated["tau_cls"]),
                "tau_e4_keep": float(gated["tau_e4_keep"]),
            }
    return dict(DEFAULT_GATED_THRESHOLDS)


class HybridE4SpecialistIDS(BaseIDSArchitecture):
    """
    Promoted Main Track 2 Architecture:
      1. Runs frozen E4 LightGBM (`leaves=63, sqrt_balanced`).
      2. Flows classified by E4 as Flood bypass specialist untouched (1 model exec).
      3. Non-Flood / Benign flows are routed to Flood-Free Stage A (Rare-vs-NonRare, 2nd exec);
         only flows with P(Rare) >= tau_rare invoke Stage B (8-Class Rare Specialist, 3rd exec).
      4. Gated override commits Stage B's class iff P(Rare) >= tau_rare, max(P_StageB) >= tau_cls,
         and (E4 already chose a rare class OR max(P_E4) < tau_e4_keep).
    """

    def __init__(
        self,
        e4_model: Any,
        stage_a_model: Any,
        stage_b_model: Any,
        taxonomy: Literal["OP29", "ORIG34"] = "OP29",
        tau_rare: float = 0.30,
        tau_cls: float = 0.25,
        tau_e4_keep: float = 0.65,
        checkpoint_paths: Optional[Sequence[Path]] = None,
        name: Optional[str] = None,
    ) -> None:
        if taxonomy not in {"OP29", "ORIG34"}:
            raise ValueError(f"Unsupported taxonomy '{taxonomy}'; expected 'OP29' or 'ORIG34'.")
        self.taxonomy: Literal["OP29", "ORIG34"] = taxonomy
        self.class_names: List[str] = list(OP29_CLASSES if taxonomy == "OP29" else THIRTY_FOUR_CLASSES)
        self.name = name or f"Hybrid_E4_Specialist_{taxonomy}"
        self.e4_model = e4_model
        self.stage_a_model = stage_a_model
        self.stage_b_model = stage_b_model
        self.tau_rare = float(tau_rare)
        self.tau_cls = float(tau_cls)
        self.tau_e4_keep = float(tau_e4_keep)
        self.flood_idx = FLOOD_29_IDX if taxonomy == "OP29" else FLOOD_34_IDX
        self.checkpoint_paths = [Path(p) for p in (checkpoint_paths or [])]

    def predict_with_trace(self, X: np.ndarray) -> Dict[str, Any]:
        """Runs cascade inference and returns predictions, probabilities, and routing trace."""
        n = len(X)
        p_e4_34 = self.e4_model.predict_proba(X)
        p_e4_tax = p_e4_34 if self.taxonomy == "ORIG34" else (p_e4_34 @ PROJ_34_TO_29)
        e4_pred = np.argmax(p_e4_tax, axis=1)

        is_flood_bypass = np.isin(e4_pred, self.flood_idx)
        routed_mask = ~is_flood_bypass

        executions = np.ones(n, dtype=np.int64)
        p_a = np.zeros(n, dtype=np.float32)
        p_b = np.zeros((n, len(RARE_8_CLASSES)), dtype=np.float32)

        if np.any(routed_mask):
            routed_idx = np.flatnonzero(routed_mask)
            executions[routed_idx] = 2  # E4 + Stage A
            p_a_r = self.stage_a_model.predict_proba(X[routed_idx])[:, 1]
            p_a[routed_idx] = p_a_r

            cand_sub = p_a_r >= self.tau_rare
            if np.any(cand_sub):
                cand_idx = routed_idx[cand_sub]
                executions[cand_idx] = 3  # E4 + Stage A + Stage B
                p_b[cand_idx] = self.stage_b_model.predict_proba(X[cand_idx])

        y_pred, p_hybrid, diagnostics = apply_gated_hybrid_policy(
            p_e4_tax=p_e4_tax,
            p_stage_a_rare=p_a,
            p_stage_b_8=p_b,
            taxonomy=self.taxonomy,
            tau_rare=self.tau_rare,
            tau_cls=self.tau_cls,
            tau_e4_keep=self.tau_e4_keep,
        )

        b_top_local = np.argmax(p_b, axis=1)
        b_top_conf = p_b[np.arange(n), b_top_local]
        override_applied = y_pred != e4_pred

        return {
            "y_pred": y_pred.astype(np.int64, copy=False),
            "y_prob": p_hybrid,
            "executions_per_flow": executions,
            "e4_pred": e4_pred.astype(np.int64, copy=False),
            "flood_bypass": is_flood_bypass,
            "routed_to_specialist": routed_mask,
            "stage_a_rare_prob": p_a,
            "stage_b_top_idx": b_top_local.astype(np.int64, copy=False),
            "stage_b_top_conf": b_top_conf.astype(np.float32, copy=False),
            "override_applied": override_applied,
            "diagnostics": diagnostics,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_with_trace(X)["y_pred"]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.predict_with_trace(X)["y_prob"]

    def model_size_bytes(self) -> int:
        unique = {p.resolve() for p in self.checkpoint_paths if p.exists()}
        return sum(p.stat().st_size for p in unique)


def load_promoted_hybrid_ids(
    taxonomy: Literal["OP29", "ORIG34"] = "OP29",
    policy_path: Optional[Path] = None,
    with_preprocessor: bool = True,
) -> Tuple[HybridE4SpecialistIDS, Optional[Any]]:
    """Loads the promoted E4 + Exp3 Hybrid Specialist IDS (and optional Track 1 preprocessor)."""
    required = [DEFAULT_E4_PATH, DEFAULT_STAGE_A_PATH, DEFAULT_STAGE_B_PATH]
    if with_preprocessor:
        required.append(DEFAULT_PREPROCESSOR_PATH)
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing checkpoint(s) required for the promoted Track 2 Hybrid IDS: "
            f"{missing}. Re-run hierarchical_ids train_and_tune_val and minority_optimization/scripts/run_exp3_specialist.py."
        )

    thresholds = load_gated_policy_thresholds(taxonomy=taxonomy, policy_path=policy_path)
    model = HybridE4SpecialistIDS(
        e4_model=joblib.load(DEFAULT_E4_PATH),
        stage_a_model=joblib.load(DEFAULT_STAGE_A_PATH),
        stage_b_model=joblib.load(DEFAULT_STAGE_B_PATH),
        taxonomy=taxonomy,
        tau_rare=thresholds["tau_rare"],
        tau_cls=thresholds["tau_cls"],
        tau_e4_keep=thresholds["tau_e4_keep"],
        checkpoint_paths=[DEFAULT_E4_PATH, DEFAULT_STAGE_A_PATH, DEFAULT_STAGE_B_PATH],
    )
    preprocessor = joblib.load(DEFAULT_PREPROCESSOR_PATH) if with_preprocessor else None
    return model, preprocessor


def preprocess_raw_flows(df: pd.DataFrame, preprocessor: Any) -> np.ndarray:
    """Applies exact Track 1 leak-free cleaning (+inf/-inf/NaN + 39-col order) and StandardScaler."""
    missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Input DataFrame missing canonical CICIoT2023 feature(s): {missing_cols}")
    X_clean = preprocessor.clean_transform(df[FEATURE_COLUMNS])
    return preprocessor.transform_scaler(X_clean)
