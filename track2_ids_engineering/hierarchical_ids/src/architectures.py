"""
Unified IDS Architecture Implementations for E1, E2, E3, and E4.

All architectures accept pre-cleaned & StandardScaler-transformed float32 features
from Track 1's Strategy 1 LeakFreePreprocessor (`X` shape `(N, 39)`) and expose a
consistent 34-class interface (`THIRTY_FOUR_CLASSES`, indices `0..33`):
  - `predict(X)` -> integer predictions in `0..33`
  - `predict_proba(X)` -> probability matrix `(N, 34)` aligned with `THIRTY_FOUR_CLASSES`
  - `predict_with_trace(X)` -> predictions, probabilities, routing traces & executions/flow
"""

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression

from track1_dataset_evaluation.src.config import (
    EIGHT_CLASSES,
    EIGHT_TO_IDX,
    THIRTY_FOUR_CLASSES,
)
from .data_utils import (
    ATTACK_CATEGORIES_7,
    BENIGN_GLOBAL_IDX,
    FLOOD_CATEGORIES,
    FLOOD_GLOBAL_INDICES,
    FLOOD_LOCAL_TO_GLOBAL,
    NON_FLOOD_GLOBAL_INDICES,
    NON_FLOOD_LOCAL_TO_GLOBAL,
)

# Precomputed index arrays for vectorized routing & probability assembly
_FLOOD_GLOBAL_ARR = np.asarray(FLOOD_GLOBAL_INDICES, dtype=np.int64)
_NON_FLOOD_GLOBAL_ARR = np.asarray(NON_FLOOD_GLOBAL_INDICES, dtype=np.int64)

_FLOOD_LOCAL_TO_GLOBAL_ARR = np.asarray(
    [FLOOD_LOCAL_TO_GLOBAL[i] for i in range(len(FLOOD_GLOBAL_INDICES))], dtype=np.int64
)
_NON_FLOOD_LOCAL_TO_GLOBAL_ARR = np.asarray(
    [NON_FLOOD_LOCAL_TO_GLOBAL[i] for i in range(len(NON_FLOOD_GLOBAL_INDICES))], dtype=np.int64
)

# Category indices considered "Flood" under both 8-class and 7-attack-class Stage 2 encoders
_FLOOD_CAT8_INDICES = {EIGHT_TO_IDX[c] for c in FLOOD_CATEGORIES}
_BENIGN_CAT8_IDX = EIGHT_TO_IDX["Benign"]
_ATTACK_CAT8_INDICES = np.asarray(
    [EIGHT_TO_IDX[c] for c in ATTACK_CATEGORIES_7], dtype=np.int64
)
_FLOOD_CAT7_INDICES = {
    i for i, c in enumerate(ATTACK_CATEGORIES_7) if c in FLOOD_CATEGORIES
}


def _safe_probs(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Clamps probabilities to avoid log(0) while preserving row normalization."""
    p = np.clip(probs.astype(np.float64, copy=False), eps, 1.0)
    p /= p.sum(axis=1, keepdims=True)
    return p


def apply_temperature_scaling(probs: np.ndarray, temperature: float) -> np.ndarray:
    """
    Applies post-hoc temperature scaling to probability distributions via log-probabilities:
        P_cal(y | x) = softmax(log(P(y | x)) / T)
    Strictly monotonic in log-probs and preserves class ranking inside the model.
    """
    if abs(temperature - 1.0) < 1e-6:
        return _safe_probs(probs)
    log_p = np.log(_safe_probs(probs)) / float(temperature)
    log_p -= log_p.max(axis=1, keepdims=True)
    exp_p = np.exp(log_p)
    return exp_p / exp_p.sum(axis=1, keepdims=True)


def fit_temperature_on_val(probs: np.ndarray, y_true: np.ndarray) -> float:
    """
    Fits scalar temperature T > 0 strictly on validation data by minimizing NLL.
    """
    log_p = np.log(_safe_probs(probs))
    rows = np.arange(len(y_true))

    def _nll(temp: float) -> float:
        scaled = log_p / temp
        scaled -= scaled.max(axis=1, keepdims=True)
        lse = np.log(np.exp(scaled).sum(axis=1))
        nll = -(scaled[rows, y_true] - lse).mean()
        return float(nll)

    res = minimize_scalar(_nll, bounds=(0.1, 10.0), method="bounded")
    return float(res.x)


def _compute_specialist_meta_features(
    p_flood: np.ndarray,
    p_non_flood: np.ndarray,
) -> np.ndarray:
    """
    Builds meta-features solely from Stage 3A and Stage 3B probability outputs
    (never uses raw input features or Stage 2 predictions).
    """
    pf = _safe_probs(p_flood)
    pn = _safe_probs(p_non_flood)

    max_f = pf.max(axis=1, keepdims=True)
    max_n = pn.max(axis=1, keepdims=True)

    sorted_f = np.sort(pf, axis=1)
    sorted_n = np.sort(pn, axis=1)
    margin_f = (sorted_f[:, -1:] - sorted_f[:, -2:-1])
    margin_n = (sorted_n[:, -1:] - sorted_n[:, -2:-1])

    ent_f = -np.sum(pf * np.log(pf), axis=1, keepdims=True) / np.log(pf.shape[1])
    ent_n = -np.sum(pn * np.log(pn), axis=1, keepdims=True) / np.log(pn.shape[1])

    return np.hstack(
        [
            np.log(max_f),
            np.log(max_n),
            max_f - max_n,
            margin_f,
            margin_n,
            ent_f,
            ent_n,
            pf,
            pn,
        ]
    ).astype(np.float32)


def _sum_file_sizes(paths: Sequence[Optional[Path]]) -> int:
    total = 0
    seen = set()
    for p in paths:
        if p is not None:
            rp = Path(p).resolve()
            if rp not in seen and rp.exists():
                seen.add(rp)
                total += rp.stat().st_size
    return total


class BaseIDSArchitecture:
    """Common interface for all evaluated 34-class IDS architectures."""

    name: str = "BaseIDS"

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_with_trace(self, X: np.ndarray) -> Dict[str, Any]:
        raise NotImplementedError

    def model_size_bytes(self) -> int:
        return 0


class Flat34BaselineArchitecture(BaseIDSArchitecture):
    """
    E1 / E4 Flat 34-Class Architecture wrapper.
    Supports optional validation-fitted temperature scaling and per-class prior/logit adjustment.
    """

    def __init__(
        self,
        name: str,
        model: Any,
        checkpoint_path: Optional[Path] = None,
        temperature: float = 1.0,
        class_logit_bias: Optional[np.ndarray] = None,
    ):
        self.name = name
        self.model = model
        self.checkpoint_path = checkpoint_path
        self.temperature = float(temperature)
        self.class_logit_bias = (
            np.asarray(class_logit_bias, dtype=np.float64)
            if class_logit_bias is not None
            else None
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw_probs = self.model.predict_proba(X)
        if abs(self.temperature - 1.0) > 1e-6 or self.class_logit_bias is not None:
            log_p = np.log(_safe_probs(raw_probs)) / self.temperature
            if self.class_logit_bias is not None:
                log_p = log_p + self.class_logit_bias[None, :]
            log_p -= log_p.max(axis=1, keepdims=True)
            exp_p = np.exp(log_p)
            return (exp_p / exp_p.sum(axis=1, keepdims=True)).astype(np.float32)
        return raw_probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        if abs(self.temperature - 1.0) < 1e-6 and self.class_logit_bias is None:
            return self.model.predict(X).astype(np.int64, copy=False)
        return np.argmax(self.predict_proba(X), axis=1).astype(np.int64)

    def predict_with_trace(self, X: np.ndarray) -> Dict[str, Any]:
        probs = self.predict_proba(X)
        preds = np.argmax(probs, axis=1).astype(np.int64)
        return {
            "y_pred": preds,
            "y_prob": probs,
            "executions_per_flow": np.ones(len(X), dtype=np.int64),
        }

    def model_size_bytes(self) -> int:
        return _sum_file_sizes([self.checkpoint_path])


class HierarchicalCascade3Stage(BaseIDSArchitecture):
    """
    E2: Three-Stage Hierarchical Cascade IDS.
      Stage 1: Binary Benign vs. Attack (if P(Attack) < attack_threshold -> return BENIGN)
      Stage 2: Category router (8-class or malicious-only 7-attack-class)
      Stage 3A: Flood Specialist (16 Flood classes -> mapped to 0..33)
      Stage 3B: Non-Flood Specialist (17 Non-Flood classes -> mapped to 0..33)
    """

    def __init__(
        self,
        name: str,
        stage1_model: Any,
        stage2_model: Any,
        flood_model: Any,
        non_flood_model: Any,
        stage2_mode: Literal["cat8", "cat7_attack"] = "cat7_attack",
        attack_threshold: float = 0.50,
        flood_route_threshold: Optional[float] = None,
        stage2_benign_policy: Literal["renormalize_attack", "exit_benign"] = "renormalize_attack",
        uncertain_margin: float = 0.60,
        checkpoint_paths: Optional[Sequence[Optional[Path]]] = None,
    ):
        self.name = name
        self.stage1_model = stage1_model
        self.stage2_model = stage2_model
        self.flood_model = flood_model
        self.non_flood_model = non_flood_model
        self.stage2_mode = stage2_mode
        self.attack_threshold = float(attack_threshold)
        self.flood_route_threshold = (
            float(flood_route_threshold) if flood_route_threshold is not None else None
        )
        self.stage2_benign_policy = stage2_benign_policy
        self.uncertain_margin = float(uncertain_margin)
        self.checkpoint_paths = list(checkpoint_paths) if checkpoint_paths else []

    def _route_stage2(self, p_s2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Determines Stage 2 category predictions and Flood vs. Non-Flood routing mask.
        Explicitly handles invalid/Benign predictions when `stage2_mode == 'cat8'`.
        """
        n = len(p_s2)
        invalid_benign = np.zeros(n, dtype=bool)
        s2_benign_exit = np.zeros(n, dtype=bool)

        if self.stage2_mode == "cat8":
            raw_cat = np.argmax(p_s2, axis=1)
            invalid_benign = raw_cat == _BENIGN_CAT8_IDX
            if self.stage2_benign_policy == "exit_benign":
                s2_benign_exit = invalid_benign.copy()
                cat_pred = raw_cat
            else:
                # Renormalize strictly over the 7 attack categories so Stage 1's Attack decision holds
                p_att = p_s2[:, _ATTACK_CAT8_INDICES].copy()
                p_att_sum = p_att.sum(axis=1, keepdims=True)
                p_att = np.where(p_att_sum > 0, p_att / np.maximum(p_att_sum, 1e-12), 1.0 / 7.0)
                best_att_local = np.argmax(p_att, axis=1)
                cat_pred = np.where(invalid_benign, _ATTACK_CAT8_INDICES[best_att_local], raw_cat)

            # Sum P(DDoS) + P(DoS) over attack categories
            p_ddos = p_s2[:, EIGHT_TO_IDX["DDoS"]]
            p_dos = p_s2[:, EIGHT_TO_IDX["DoS"]]
            p_att_total = np.maximum(1.0 - p_s2[:, _BENIGN_CAT8_IDX], 1e-12)
            p_flood = np.clip((p_ddos + p_dos) / p_att_total, 0.0, 1.0)

            if self.flood_route_threshold is not None:
                route_flood = (~s2_benign_exit) & (p_flood >= self.flood_route_threshold)
            else:
                route_flood = np.isin(cat_pred, list(_FLOOD_CAT8_INDICES)) & (~s2_benign_exit)
        else:
            # cat7_attack: classes are ATTACK_CATEGORIES_7 (indices 0..6)
            cat_local = np.argmax(p_s2, axis=1)
            cat_pred = _ATTACK_CAT8_INDICES[cat_local]
            p_flood = np.clip(
                sum(p_s2[:, idx] for idx in _FLOOD_CAT7_INDICES),
                0.0,
                1.0,
            )
            if self.flood_route_threshold is not None:
                route_flood = p_flood >= self.flood_route_threshold
            else:
                route_flood = np.isin(cat_local, list(_FLOOD_CAT7_INDICES))

        return cat_pred, p_flood, route_flood, invalid_benign, s2_benign_exit

    def predict_with_trace(self, X: np.ndarray) -> Dict[str, Any]:
        n = len(X)
        p_bin = self.stage1_model.predict_proba(X)
        p_attack = p_bin[:, 1]
        s1_is_attack = p_attack >= self.attack_threshold
        s1_uncertain = np.maximum(p_bin[:, 0], p_attack) < self.uncertain_margin

        y_pred = np.full(n, BENIGN_GLOBAL_IDX, dtype=np.int64)
        y_prob = np.zeros((n, len(THIRTY_FOUR_CLASSES)), dtype=np.float32)
        executions = np.ones(n, dtype=np.int64)

        # Benign exits from Stage 1
        benign_mask = ~s1_is_attack
        if np.any(benign_mask):
            # Distribute tiny residual across attack classes so row sums to 1 and argmax is always BENIGN
            p_ben = np.clip(p_bin[benign_mask, 0], 1e-6, 1.0 - 1e-6)
            # Ensure BENIGN remains strict argmax even if attack_threshold < 0.50
            eff_ben = np.maximum(p_ben, 0.51)
            rem = (1.0 - eff_ben) / 33.0
            y_prob[benign_mask, :] = rem[:, None]
            y_prob[benign_mask, BENIGN_GLOBAL_IDX] = eff_ben

        stage2_cat_pred = np.full(n, -1, dtype=np.int64)
        stage2_prob_flood = np.full(n, np.nan, dtype=np.float32)
        stage2_invalid_benign = np.zeros(n, dtype=bool)
        stage2_uncertain = np.zeros(n, dtype=bool)
        specialist_route = np.full(n, "BENIGN_EXIT", dtype=object)

        if np.any(s1_is_attack):
            att_idx = np.flatnonzero(s1_is_attack)
            X_att = X[att_idx]
            p_s2 = self.stage2_model.predict_proba(X_att)
            executions[att_idx] += 1

            cat_pred, p_flood, route_flood, inv_ben, s2_ben_exit = self._route_stage2(p_s2)
            stage2_cat_pred[att_idx] = cat_pred
            stage2_prob_flood[att_idx] = p_flood.astype(np.float32)
            stage2_invalid_benign[att_idx] = inv_ben
            stage2_uncertain[att_idx] = p_s2.max(axis=1) < self.uncertain_margin

            if np.any(s2_ben_exit):
                b_idx = att_idx[s2_ben_exit]
                y_pred[b_idx] = BENIGN_GLOBAL_IDX
                y_prob[b_idx, :] = 1e-6
                y_prob[b_idx, BENIGN_GLOBAL_IDX] = 1.0 - 33e-6
                specialist_route[b_idx] = "STAGE2_BENIGN_EXIT"

            flood_idx = att_idx[route_flood]
            if len(flood_idx) > 0:
                p_3a = self.flood_model.predict_proba(X[flood_idx])
                executions[flood_idx] += 1
                local_pred_3a = np.argmax(p_3a, axis=1)
                y_pred[flood_idx] = _FLOOD_LOCAL_TO_GLOBAL_ARR[local_pred_3a]
                y_prob[np.ix_(flood_idx, _FLOOD_GLOBAL_ARR)] = p_3a.astype(np.float32)
                specialist_route[flood_idx] = "3A_FLOOD"

            non_flood_mask = (~route_flood) & (~s2_ben_exit)
            non_flood_idx = att_idx[non_flood_mask]
            if len(non_flood_idx) > 0:
                p_3b = self.non_flood_model.predict_proba(X[non_flood_idx])
                executions[non_flood_idx] += 1
                local_pred_3b = np.argmax(p_3b, axis=1)
                y_pred[non_flood_idx] = _NON_FLOOD_LOCAL_TO_GLOBAL_ARR[local_pred_3b]
                y_prob[np.ix_(non_flood_idx, _NON_FLOOD_GLOBAL_ARR)] = p_3b.astype(np.float32)
                specialist_route[non_flood_idx] = "3B_NON_FLOOD"

        return {
            "y_pred": y_pred,
            "y_prob": y_prob,
            "executions_per_flow": executions,
            "stage1_pred_attack": s1_is_attack,
            "stage1_prob_attack": p_attack.astype(np.float32),
            "stage1_uncertain": s1_uncertain,
            "stage2_pred_cat": stage2_cat_pred,
            "stage2_prob_flood": stage2_prob_flood,
            "stage2_invalid_benign": stage2_invalid_benign,
            "stage2_uncertain": stage2_uncertain,
            "specialist_route": specialist_route,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_with_trace(X)["y_pred"]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.predict_with_trace(X)["y_prob"]

    def model_size_bytes(self) -> int:
        return _sum_file_sizes(self.checkpoint_paths)


class ParallelSpecialistFusion(BaseIDSArchitecture):
    """
    E3: Parallel Specialist Architecture (No Stage 2 category classifier).
      Stage 1: Binary Benign vs. Attack (if P(Attack) < attack_threshold -> return BENIGN)
      Stage 3A || Stage 3B: Flood specialist and Non-Flood specialist run on all Attacks
      Fusion modes:
        - 'naive_max': Uncalibrated direct comparison max(P_3A, P_3B)
        - 'temp_alpha': Validation temperature-scaled specialists weighted by scalar alpha
                        (P_flood_scaled = alpha * P_3A^(T_A), P_nonflood = (1-alpha) * P_3B^(T_B))
        - 'meta_gate_lr': Validation-fitted Logistic Regression meta-gate taking ONLY
                          calibrated Stage 3A & 3B probability summaries/distributions to
                          estimate g(x) = P(Flood | specialist outputs) and fuse:
                          [g(x) * P_3A^(T_A), (1 - g(x)) * P_3B^(T_B)]
    """

    def __init__(
        self,
        name: str,
        stage1_model: Any,
        flood_model: Any,
        non_flood_model: Any,
        fusion_mode: Literal["naive_max", "temp_alpha", "meta_gate_lr"] = "temp_alpha",
        attack_threshold: float = 0.50,
        temp_flood: float = 1.0,
        temp_non_flood: float = 1.0,
        alpha_flood: float = 0.50,
        meta_gate_model: Optional[LogisticRegression] = None,
        checkpoint_paths: Optional[Sequence[Optional[Path]]] = None,
    ):
        self.name = name
        self.stage1_model = stage1_model
        self.flood_model = flood_model
        self.non_flood_model = non_flood_model
        self.fusion_mode = fusion_mode
        self.attack_threshold = float(attack_threshold)
        self.temp_flood = float(temp_flood)
        self.temp_non_flood = float(temp_non_flood)
        self.alpha_flood = float(alpha_flood)
        self.meta_gate_model = meta_gate_model
        self.checkpoint_paths = list(checkpoint_paths) if checkpoint_paths else []

    def fuse_specialist_probs(
        self,
        p_3a_raw: np.ndarray,
        p_3b_raw: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fuses Stage 3A (N, 16) and Stage 3B (N, 17) probabilities into joint attack
        probabilities `(N, 34)` (with Benign=0 before Stage 1 combining) and returns
        the family gate weight `g_flood` and disagreement indicators.
        """
        n = len(p_3a_raw)
        if self.fusion_mode == "naive_max":
            p_3a = _safe_probs(p_3a_raw)
            p_3b = _safe_probs(p_3b_raw)
            # Raw direct comparison: stack and normalize across all 33 attack classes
            joint = np.zeros((n, len(THIRTY_FOUR_CLASSES)), dtype=np.float64)
            joint[:, _FLOOD_GLOBAL_ARR] = p_3a
            joint[:, _NON_FLOOD_GLOBAL_ARR] = p_3b
            joint /= joint.sum(axis=1, keepdims=True)
            g_flood = (p_3a.max(axis=1) >= p_3b.max(axis=1)).astype(np.float32)
        elif self.fusion_mode == "temp_alpha":
            p_3a = apply_temperature_scaling(p_3a_raw, self.temp_flood)
            p_3b = apply_temperature_scaling(p_3b_raw, self.temp_non_flood)
            joint = np.zeros((n, len(THIRTY_FOUR_CLASSES)), dtype=np.float64)
            joint[:, _FLOOD_GLOBAL_ARR] = self.alpha_flood * p_3a
            joint[:, _NON_FLOOD_GLOBAL_ARR] = (1.0 - self.alpha_flood) * p_3b
            g_flood = joint[:, _FLOOD_GLOBAL_ARR].sum(axis=1).astype(np.float32)
        elif self.fusion_mode == "meta_gate_lr":
            if self.meta_gate_model is None:
                raise RuntimeError("meta_gate_lr mode requires a fitted meta_gate_model.")
            p_3a = apply_temperature_scaling(p_3a_raw, self.temp_flood)
            p_3b = apply_temperature_scaling(p_3b_raw, self.temp_non_flood)
            meta_X = _compute_specialist_meta_features(p_3a, p_3b)
            g_flood = self.meta_gate_model.predict_proba(meta_X)[:, 1].astype(np.float64)
            joint = np.zeros((n, len(THIRTY_FOUR_CLASSES)), dtype=np.float64)
            joint[:, _FLOOD_GLOBAL_ARR] = g_flood[:, None] * p_3a
            joint[:, _NON_FLOOD_GLOBAL_ARR] = (1.0 - g_flood[:, None]) * p_3b
        else:
            raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")

        # Specialist disagreement flags:
        # 1) Both specialists confident (>= 0.50 in their closed world)
        # 2) Naive uncalibrated family winner disagrees with fused family winner
        both_confident = (p_3a_raw.max(axis=1) >= 0.50) & (p_3b_raw.max(axis=1) >= 0.50)
        return joint.astype(np.float32), g_flood.astype(np.float32), both_confident

    def predict_with_trace(self, X: np.ndarray) -> Dict[str, Any]:
        n = len(X)
        p_bin = self.stage1_model.predict_proba(X)
        p_attack = p_bin[:, 1]
        s1_is_attack = p_attack >= self.attack_threshold

        y_pred = np.full(n, BENIGN_GLOBAL_IDX, dtype=np.int64)
        y_prob = np.zeros((n, len(THIRTY_FOUR_CLASSES)), dtype=np.float32)
        executions = np.ones(n, dtype=np.int64)

        benign_mask = ~s1_is_attack
        if np.any(benign_mask):
            p_ben = np.clip(p_bin[benign_mask, 0], 1e-6, 1.0 - 1e-6)
            eff_ben = np.maximum(p_ben, 0.51)
            rem = (1.0 - eff_ben) / 33.0
            y_prob[benign_mask, :] = rem[:, None]
            y_prob[benign_mask, BENIGN_GLOBAL_IDX] = eff_ben

        gate_flood_prob = np.full(n, np.nan, dtype=np.float32)
        both_confident = np.zeros(n, dtype=bool)
        naive_vs_fused_disagree = np.zeros(n, dtype=bool)
        top_3a_global = np.full(n, -1, dtype=np.int64)
        top_3b_global = np.full(n, -1, dtype=np.int64)
        top_3a_conf = np.full(n, np.nan, dtype=np.float32)
        top_3b_conf = np.full(n, np.nan, dtype=np.float32)

        if np.any(s1_is_attack):
            att_idx = np.flatnonzero(s1_is_attack)
            X_att = X[att_idx]
            p_3a_raw = self.flood_model.predict_proba(X_att)
            p_3b_raw = self.non_flood_model.predict_proba(X_att)
            executions[att_idx] += 2  # Both specialists executed

            joint_att, g_flood, both_conf = self.fuse_specialist_probs(p_3a_raw, p_3b_raw)
            fused_pred = np.argmax(joint_att, axis=1)

            y_pred[att_idx] = fused_pred
            y_prob[att_idx] = joint_att
            gate_flood_prob[att_idx] = g_flood
            both_confident[att_idx] = both_conf

            top_3a_local = np.argmax(p_3a_raw, axis=1)
            top_3b_local = np.argmax(p_3b_raw, axis=1)
            top_3a_global[att_idx] = _FLOOD_LOCAL_TO_GLOBAL_ARR[top_3a_local]
            top_3b_global[att_idx] = _NON_FLOOD_LOCAL_TO_GLOBAL_ARR[top_3b_local]
            top_3a_conf[att_idx] = p_3a_raw.max(axis=1).astype(np.float32)
            top_3b_conf[att_idx] = p_3b_raw.max(axis=1).astype(np.float32)

            naive_flood_wins = p_3a_raw.max(axis=1) >= p_3b_raw.max(axis=1)
            fused_flood_wins = np.isin(fused_pred, _FLOOD_GLOBAL_ARR)
            naive_vs_fused_disagree[att_idx] = naive_flood_wins != fused_flood_wins

        return {
            "y_pred": y_pred,
            "y_prob": y_prob,
            "executions_per_flow": executions,
            "stage1_pred_attack": s1_is_attack,
            "stage1_prob_attack": p_attack.astype(np.float32),
            "gate_flood_prob": gate_flood_prob,
            "both_specialists_confident": both_confident,
            "naive_vs_fused_disagree": naive_vs_fused_disagree,
            "top_3a_global": top_3a_global,
            "top_3b_global": top_3b_global,
            "top_3a_conf": top_3a_conf,
            "top_3b_conf": top_3b_conf,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_with_trace(X)["y_pred"]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.predict_with_trace(X)["y_prob"]

    def model_size_bytes(self) -> int:
        return _sum_file_sizes(self.checkpoint_paths)
