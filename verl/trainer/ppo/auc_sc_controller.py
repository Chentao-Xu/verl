"""Exploration-retaining, power-tempered token-importance sampling for GRPO."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from numbers import Real
import struct
from typing import Mapping, Optional

import torch
from torch import Tensor


_METRIC_PREFIX = "rollout_corr/exploration_tis_"
_FLOAT32_INFO = torch.finfo(torch.float32)
# These are runtime-safe IEEE-754 binary64 constants.  They are intentionally
# below the corresponding theoretical limits so float32 exponentiation and
# directed rounding remain finite and normal on every supported device.
MAX_LOG_BOUND = float.fromhex("0x1.5d589ep+6")
MAX_LOG_CAP = float.fromhex("0x1.5d589ep+5")
MAX_CAP = float.fromhex("0x1.ffffb4p+62")


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


def _positive_fp32(value: object, name: str) -> float:
    value = _finite_real(value, name)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")
    try:
        value_fp32 = struct.unpack("f", struct.pack("f", value))[0]
    except OverflowError as error:
        raise ValueError(f"{name} must remain finite and positive in FP32") from error
    if not math.isfinite(value_fp32) or value_fp32 <= 0.0:
        raise ValueError(f"{name} must remain finite and positive in FP32")
    return value_fp32


def _validate_nonnegative_fp32(value: object, name: str) -> None:
    value = _finite_real(value, name)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative")
    try:
        value_fp32 = struct.unpack("f", struct.pack("f", value))[0]
    except OverflowError as error:
        raise ValueError(f"{name} must remain finite and non-negative in FP32") from error
    if not math.isfinite(value_fp32) or (value > 0.0 and value_fp32 <= 0.0):
        raise ValueError(f"{name} must remain finite and non-negative in FP32")


@dataclass(frozen=True)
class ExplorationTISState:
    """Persistent controller state, deliberately independent of AIS state."""

    alpha: float
    previous_response_length_mean: Optional[float]
    source_step: int
    sequence_q_history: tuple[float, ...] = ()
    lower_release: float = 0.0
    mismatch_ema: Optional[float] = None
    mismatch_baseline_ema: Optional[float] = None
    mismatch_baseline_count: int = 0
    mismatch_dual_beta: float = 0.0
    mismatch_ess_baseline_ema: Optional[float] = None
    mismatch_support_baseline_ema: Optional[float] = None
    mismatch_excess_steps: int = 0

    def __post_init__(self) -> None:
        alpha = _finite_real(self.alpha, "alpha")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        lower_release = _finite_real(self.lower_release, "lower_release")
        if not 0.0 <= lower_release <= 1.0:
            raise ValueError("lower_release must be in [0, 1]")
        if self.previous_response_length_mean is not None:
            previous_length = _finite_real(self.previous_response_length_mean, "previous_response_length_mean")
            if previous_length <= 0.0:
                raise ValueError("previous_response_length_mean must be positive when present")
        if isinstance(self.source_step, bool) or not isinstance(self.source_step, int) or self.source_step < 0:
            raise ValueError("source_step must be a non-negative integer")
        if not isinstance(self.sequence_q_history, tuple):
            raise ValueError("sequence_q_history must be a tuple")
        normalized_history = []
        for index, value in enumerate(self.sequence_q_history):
            history_value = _finite_real(value, f"sequence_q_history[{index}]")
            if history_value < 0.0:
                raise ValueError("sequence_q_history entries must be non-negative")
            normalized_history.append(history_value)
        object.__setattr__(self, "sequence_q_history", tuple(normalized_history))
        for name, value in (
            ("mismatch_ema", self.mismatch_ema),
            ("mismatch_baseline_ema", self.mismatch_baseline_ema),
            ("mismatch_ess_baseline_ema", self.mismatch_ess_baseline_ema),
            ("mismatch_support_baseline_ema", self.mismatch_support_baseline_ema),
        ):
            if value is not None and _finite_real(value, name) < 0.0:
                raise ValueError(f"{name} must be non-negative when present")
        if isinstance(self.mismatch_baseline_count, bool) or not isinstance(self.mismatch_baseline_count, int):
            raise ValueError("mismatch_baseline_count must be an integer")
        if self.mismatch_baseline_count < 0:
            raise ValueError("mismatch_baseline_count must be non-negative")
        if _finite_real(self.mismatch_dual_beta, "mismatch_dual_beta") < 0.0:
            raise ValueError("mismatch_dual_beta must be non-negative")
        if isinstance(self.mismatch_excess_steps, bool) or not isinstance(self.mismatch_excess_steps, int):
            raise ValueError("mismatch_excess_steps must be an integer")
        if self.mismatch_excess_steps < 0:
            raise ValueError("mismatch_excess_steps must be non-negative")

    @classmethod
    def create(cls) -> "ExplorationTISState":
        return cls(alpha=0.0, previous_response_length_mean=None, source_step=0, sequence_q_history=())

    def state_dict(self) -> dict[str, object]:
        return {
            "alpha": self.alpha,
            "previous_response_length_mean": self.previous_response_length_mean,
            "source_step": self.source_step,
            "sequence_q_history": [float(value) for value in self.sequence_q_history],
            "lower_release": float(self.lower_release),
            "mismatch_ema": self.mismatch_ema,
            "mismatch_baseline_ema": self.mismatch_baseline_ema,
            "mismatch_baseline_count": self.mismatch_baseline_count,
            "mismatch_dual_beta": self.mismatch_dual_beta,
            "mismatch_ess_baseline_ema": self.mismatch_ess_baseline_ema,
            "mismatch_support_baseline_ema": self.mismatch_support_baseline_ema,
            "mismatch_excess_steps": self.mismatch_excess_steps,
        }

    @classmethod
    def from_state_dict(cls, state_dict: Mapping[str, object]) -> "ExplorationTISState":
        if not isinstance(state_dict, Mapping):
            raise TypeError("state_dict must be a mapping")
        legacy_keys = {"alpha", "previous_response_length_mean", "source_step"}
        new_keys = legacy_keys | {"sequence_q_history"}
        state_keys = set(state_dict)
        trust_keys = {
            "mismatch_ema", "mismatch_baseline_ema", "mismatch_baseline_count", "mismatch_dual_beta"
        }
        multigate_keys = {
            "mismatch_ess_baseline_ema", "mismatch_support_baseline_ema", "mismatch_excess_steps"
        }
        if state_keys not in (
            legacy_keys,
            new_keys,
            new_keys | {"lower_release"},
            new_keys | {"lower_release"} | trust_keys,
            new_keys | {"lower_release"} | trust_keys | multigate_keys,
        ):
            raise ValueError(
                "state_dict must contain exactly alpha, previous_response_length_mean, source_step, "
                "and optionally sequence_q_history and lower_release"
            )
        sequence_q_history: tuple[object, ...] = ()
        if "sequence_q_history" in state_keys:
            serialized_history = state_dict["sequence_q_history"]
            if isinstance(serialized_history, (str, bytes, bytearray)) or not isinstance(
                serialized_history, Sequence
            ):
                raise ValueError("sequence_q_history must be a non-string sequence")
            sequence_q_history = tuple(serialized_history)
        return cls(
            alpha=state_dict["alpha"],  # type: ignore[arg-type]
            previous_response_length_mean=state_dict["previous_response_length_mean"],  # type: ignore[arg-type]
            source_step=state_dict["source_step"],  # type: ignore[arg-type]
            sequence_q_history=sequence_q_history,  # type: ignore[arg-type]
            lower_release=state_dict.get("lower_release", 0.0),  # type: ignore[arg-type]
            mismatch_ema=state_dict.get("mismatch_ema"),  # type: ignore[arg-type]
            mismatch_baseline_ema=state_dict.get("mismatch_baseline_ema"),  # type: ignore[arg-type]
            mismatch_baseline_count=state_dict.get("mismatch_baseline_count", 0),  # type: ignore[arg-type]
            mismatch_dual_beta=state_dict.get("mismatch_dual_beta", 0.0),  # type: ignore[arg-type]
            mismatch_ess_baseline_ema=state_dict.get("mismatch_ess_baseline_ema"),  # type: ignore[arg-type]
            mismatch_support_baseline_ema=state_dict.get("mismatch_support_baseline_ema"),  # type: ignore[arg-type]
            mismatch_excess_steps=state_dict.get("mismatch_excess_steps", 0),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class ExplorationTISResult:
    weights: Tensor
    proposed_state: ExplorationTISState
    actor_update_allowed: bool
    metrics: dict[str, float]


def _validate_tensor_inputs(
    actor_old_log_probs: Tensor,
    rollout_log_probs: Tensor,
    advantages: Tensor,
    response_mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    tensors = {
        "actor_old_log_probs": actor_old_log_probs,
        "rollout_log_probs": rollout_log_probs,
        "advantages": advantages,
        "response_mask": response_mask,
    }
    if any(not isinstance(value, Tensor) for value in tensors.values()):
        raise TypeError("all inputs must be torch tensors")
    shape = actor_old_log_probs.shape
    if actor_old_log_probs.ndim != 2 or any(value.ndim != 2 or value.shape != shape for value in tensors.values()):
        raise ValueError("all inputs must have the same 2D shape")
    if shape[0] == 0 or shape[1] == 0:
        raise ValueError("all inputs must have non-empty batch and token dimensions")
    if any(value.device != actor_old_log_probs.device for value in tensors.values()):
        raise ValueError("all inputs must be on the same device")
    for name, value in (
        ("actor_old_log_probs", actor_old_log_probs),
        ("rollout_log_probs", rollout_log_probs),
        ("advantages", advantages),
    ):
        if not value.is_floating_point() or value.is_complex():
            raise TypeError(f"{name} must be a real floating tensor")
    if response_mask.dtype == torch.bool:
        valid = response_mask
        mask_semantics_valid = torch.ones((), dtype=torch.bool, device=response_mask.device)
    elif response_mask.is_complex():
        valid = response_mask != 0
        mask_semantics_valid = torch.zeros((), dtype=torch.bool, device=response_mask.device)
    else:
        valid = response_mask.bool()
        mask_semantics_valid = torch.isfinite(response_mask).all() & torch.all((response_mask == 0) | (response_mask == 1))
    return valid, mask_semantics_valid, valid.any(dim=1).all()


def _seq_token_mean(values: Tensor, valid: Tensor) -> Tensor:
    """Mean each row over valid tokens, then mean the rows without overflowing."""
    selected = torch.where(valid, values, torch.zeros_like(values))
    scale = selected.abs().amax()
    safe_scale = torch.where(scale > 0.0, scale, torch.ones_like(scale))
    safe_lengths = valid.sum(dim=1, keepdim=True).to(values.dtype).clamp_min(1.0)
    return ((selected / safe_scale).sum(dim=1, keepdim=True) / safe_lengths).mean() * safe_scale


def _conditional_weight_mean(weights: Tensor, condition: Tensor) -> Tensor:
    selected = torch.where(condition, weights, torch.zeros_like(weights))
    scale = selected.abs().amax()
    safe_scale = torch.where(scale > 0.0, scale, torch.ones_like(scale))
    count = condition.sum().to(weights.dtype)
    mean = (selected / safe_scale).sum() / count.clamp_min(1.0) * safe_scale
    return torch.where(count > 0.0, mean, torch.ones_like(mean))


def _fp32_floor(value64: Tensor) -> Tensor:
    """Round a positive float64 scalar down to float32 without leaving its device."""
    rounded32 = value64.to(torch.float32)
    zero32 = rounded32.new_zeros(())
    return torch.where(rounded32.to(torch.float64) > value64, torch.nextafter(rounded32, zero32), rounded32)


def _fp32_ceil(value64: Tensor) -> Tensor:
    """Round a positive float64 scalar up to float32 without leaving its device."""
    rounded32 = value64.to(torch.float32)
    infinity32 = rounded32.new_full((), float("inf"))
    return torch.where(rounded32.to(torch.float64) < value64, torch.nextafter(rounded32, infinity32), rounded32)


def _fp32_power_bounds(cap: float, exponent: Tensor) -> tuple[Tensor, Tensor]:
    """Return inward-directed float32 bounds for ``[cap**-exponent, cap**exponent]``."""
    exponent64 = exponent.to(torch.float64)
    cap64 = exponent64.new_ones(()) * cap
    upper = _fp32_floor(torch.pow(cap64, exponent64))
    lower = _fp32_ceil(torch.reciprocal(upper.to(torch.float64)))
    return lower, upper


def _validate_config(
    state: ExplorationTISState,
    source_step: int,
    response_length_cap: int,
    cap: float,
    half_life: float,
    log_ratio_bound: float,
    historical_sequence_risk_enabled: bool,
    sequence_history_size: int,
    sequence_min_history: int,
    sequence_log_scale_floor: float,
    sequence_risk_saturation_z: float,
) -> tuple[int, float, float, float, bool, int, int, float, float]:
    if not isinstance(state, ExplorationTISState):
        raise TypeError("state must be an ExplorationTISState")
    if isinstance(source_step, bool) or not isinstance(source_step, int) or source_step <= state.source_step:
        raise ValueError("source_step must be an integer greater than state.source_step")
    if isinstance(response_length_cap, bool) or not isinstance(response_length_cap, int) or response_length_cap <= 0:
        raise ValueError("response_length_cap must be a positive integer")
    if state.previous_response_length_mean is not None and state.previous_response_length_mean > response_length_cap:
        raise ValueError("previous_response_length_mean must not exceed response_length_cap")
    cap = _finite_real(cap, "cap")
    half_life = _finite_real(half_life, "half_life")
    log_ratio_bound = _finite_real(log_ratio_bound, "log_ratio_bound")
    if cap <= 1.0:
        raise ValueError("cap must be greater than 1")
    if half_life <= 0.0 or log_ratio_bound <= 0.0:
        raise ValueError("half_life and log_ratio_bound must be positive")
    if cap > MAX_CAP:
        raise ValueError(f"cap must not exceed the FP32-safe maximum {MAX_CAP}")
    if math.log(cap) > MAX_LOG_CAP:
        raise ValueError(f"log(cap) must not exceed the FP32-safe maximum {MAX_LOG_CAP}")
    if log_ratio_bound > MAX_LOG_BOUND:
        raise ValueError(f"log_ratio_bound must not exceed the FP32-safe maximum {MAX_LOG_BOUND}")
    if not isinstance(historical_sequence_risk_enabled, bool):
        raise TypeError("historical_sequence_risk_enabled must be a bool")
    if isinstance(sequence_history_size, bool) or not isinstance(sequence_history_size, int):
        raise TypeError("sequence_history_size must be an integer")
    if isinstance(sequence_min_history, bool) or not isinstance(sequence_min_history, int):
        raise TypeError("sequence_min_history must be an integer")
    if sequence_history_size < 1:
        raise ValueError("sequence_history_size must be at least 1")
    if not 1 <= sequence_min_history <= sequence_history_size:
        raise ValueError("sequence_min_history must be between 1 and sequence_history_size")
    sequence_log_scale_floor = _positive_fp32(sequence_log_scale_floor, "sequence_log_scale_floor")
    sequence_risk_saturation_z = _positive_fp32(sequence_risk_saturation_z, "sequence_risk_saturation_z")
    if len(state.sequence_q_history) > sequence_history_size:
        raise ValueError("sequence_q_history length must not exceed sequence_history_size")
    if not historical_sequence_risk_enabled and state.sequence_q_history:
        raise ValueError("sequence_q_history must be empty when historical sequence risk is disabled")
    for index, value in enumerate(state.sequence_q_history):
        _validate_nonnegative_fp32(value, f"sequence_q_history[{index}]")
    return (
        response_length_cap,
        cap,
        half_life,
        log_ratio_bound,
        historical_sequence_risk_enabled,
        sequence_history_size,
        sequence_min_history,
        sequence_log_scale_floor,
        sequence_risk_saturation_z,
    )


def _historical_sequence_risk(
    current_q: Tensor,
    sequence_q_history: tuple[float, ...],
    sequence_min_history: int,
    sequence_log_scale_floor: float,
    sequence_risk_saturation_z: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Return the past-only historical risk and its scalar FP32 telemetry."""
    zero = current_q.new_zeros(())
    if len(sequence_q_history) < sequence_min_history:
        return zero, zero, zero, zero, zero, zero, zero

    one = current_q.new_ones(())
    eps = one * _FLOAT32_INFO.eps
    if current_q.device.type == "cuda":
        history = torch.tensor(sequence_q_history, dtype=torch.float32, device="cpu", pin_memory=True).to(
            device=current_q.device, non_blocking=True
        )
    else:
        history = current_q.new_tensor(sequence_q_history, dtype=torch.float32)
    logs = torch.log(history + eps)
    median = torch.quantile(logs, 0.5, interpolation="linear")
    q25 = torch.quantile(logs, 0.25, interpolation="linear")
    q75 = torch.quantile(logs, 0.75, interpolation="linear")
    iqr_scale = (q75 - q25) / 1.349
    mad_scale = 1.4826 * torch.quantile((logs - median).abs(), 0.5, interpolation="linear")
    scale_floor = one * sequence_log_scale_floor
    scale = torch.maximum(torch.maximum(iqr_scale, mad_scale), scale_floor)
    z_positive = torch.nan_to_num(
        torch.clamp_min((torch.log(current_q + eps) - median) / scale, 0.0),
        nan=0.0,
        posinf=_FLOAT32_INFO.max,
        neginf=0.0,
    )
    saturation_z = one * sequence_risk_saturation_z
    risk = torch.clamp(z_positive / saturation_z, min=0.0, max=1.0)
    reference = torch.nan_to_num(
        torch.clamp(torch.exp(median) - eps, min=0.0),
        nan=0.0,
        posinf=_FLOAT32_INFO.max,
        neginf=0.0,
    )
    severe_incident = (z_positive >= saturation_z).to(torch.float32)
    return risk, one, reference, median, scale, z_positive, severe_incident


def _apply_low_residual_budget(raw_log_ratio, advantages, valid, alpha, upper_weights, beta):
    """Full-batch seq-mean-token-mean budget; no epsilon grants free residual.

    FP64 reductions avoid overflow from finite FP32 advantages. The upper-side
    tensor is reused exactly. This bounds a detached contribution, not parameter
    gradients or the quality of the reference update.
    """
    logs = raw_log_ratio.to(torch.float64).clamp_max(0.0)
    low = valid & (raw_log_ratio < 0.0)
    rho = logs.exp()
    candidate = torch.where(alpha == 0.0, torch.ones_like(logs), (alpha.double() * logs).exp())
    magnitude = advantages.double().abs()
    lengths = valid.sum(1).double().clamp_min(1.0)

    def reduce(values):
        return (torch.where(low, values, 0.0).sum(1) / lengths).mean()

    reference = magnitude * rho
    residual = magnitude * (candidate - rho).clamp_min(0.0)
    s, d = reduce(reference), reduce(residual)
    if beta == 0.0:
        coefficient = s.new_zeros(())
    else:
        coefficient = torch.where(d > 0.0, (beta * s / torch.where(d > 0.0, d, 1.0)).clamp_max(1.0), 1.0)
    weights = torch.where(low, (rho + coefficient * (candidate - rho)).float(), upper_weights)
    # Measure the actually stored FP32 weights, including rounding error.
    applied = magnitude * (weights.double() - rho).clamp_min(0.0)
    applied_d = reduce(applied)
    metrics = {
        "enabled": s.new_ones(()), "S": s, "D": d, "applied_D": applied_d,
        "lambda": coefficient, "beta": s.new_tensor(beta),
        "budget_binding": (coefficient < 1.0).double(),
        "budget_slack": beta * s - applied_d,
        "rho_underflow_fraction": reduce((rho.float() == 0.0).double()),
    }
    for label, selection in (("positive", advantages > 0), ("negative", advantages < 0)):
        metrics[label + "_S"] = reduce(reference * selection)
        metrics[label + "_D"] = reduce(residual * selection)
        metrics[label + "_applied_D"] = reduce(applied * selection)
    return weights, metrics


def compute_exploration_tis(
    actor_old_log_probs: Tensor,
    rollout_log_probs: Tensor,
    advantages: Tensor,
    response_mask: Tensor,
    *,
    state: ExplorationTISState,
    source_step: int,
    response_length_cap: int,
    cap: float = 5.0,
    weight_mode: str = "legacy",
    half_life: float = 8.0,
    log_ratio_bound: float = 20.0,
    historical_sequence_risk_enabled: bool = False,
    sequence_history_size: int = 32,
    sequence_min_history: int = 8,
    sequence_log_scale_floor: float = 0.05,
    sequence_risk_saturation_z: float = 4.0,
    apply_control: bool = True,
    length_risk_enabled: bool = False,
    adaptive_lower_enabled: bool = False,
    lower_max_log_bound: float = 20.0,
    low_residual_budget_enabled: bool = False,
    low_residual_budget_beta: float = 0.1,
    mismatch_trust_region_enabled: bool = False,
    mismatch_trust_warmup_start_step: int = 20,
    mismatch_trust_warmup_end_step: int = 80,
    mismatch_trust_baseline_multiplier: float = 1.25,
    mismatch_trust_ema_decay: float = 0.8,
    mismatch_trust_dual_lr: float = 0.02,
    mismatch_trust_dual_beta_max: float = 1.0,
    mismatch_trust_pg_scale_min: float = 0.25,
    mismatch_trust_activation_ratio: float = 1.15,
    mismatch_trust_persistence_steps: int = 8,
    mismatch_trust_pg_activation_ratio: float = 1.5,
    mismatch_trust_ess_drop_threshold: float = 0.05,
    mismatch_trust_support_drop_threshold: float = 0.10,
    mismatch_trust_length_risk_threshold: float = 0.05,
) -> ExplorationTISResult:
    """Compute sequence-risk telemetry and optionally apply detached tempered weights."""
    if not isinstance(apply_control, bool):
        raise TypeError("apply_control must be a bool")
    if not isinstance(length_risk_enabled, bool):
        raise TypeError("length_risk_enabled must be a bool")
    if not isinstance(adaptive_lower_enabled, bool):
        raise TypeError("adaptive_lower_enabled must be a bool")
    if not isinstance(low_residual_budget_enabled, bool):
        raise TypeError("low_residual_budget_enabled must be a bool")
    if weight_mode not in ("legacy", "symmetric_power", "lower_linear", "positive_lower_power"):
        raise ValueError("unknown exploration TIS weight_mode")
    if weight_mode != "legacy" and (adaptive_lower_enabled or low_residual_budget_enabled):
        raise ValueError("new weight modes require adaptive_lower and low_residual_budget disabled")
    if not isinstance(mismatch_trust_region_enabled, bool):
        raise TypeError("mismatch_trust_region_enabled must be a bool")
    low_residual_budget_beta = _finite_real(low_residual_budget_beta, "low_residual_budget_beta")
    if low_residual_budget_beta < 0.0:
        raise ValueError("low_residual_budget_beta must be nonnegative")
    if low_residual_budget_enabled and (adaptive_lower_enabled or not apply_control):
        raise ValueError("low_residual_budget requires apply_control and is mutually exclusive with adaptive_lower")
    for name, value in (
        ("mismatch_trust_warmup_start_step", mismatch_trust_warmup_start_step),
        ("mismatch_trust_warmup_end_step", mismatch_trust_warmup_end_step),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if mismatch_trust_warmup_end_step < mismatch_trust_warmup_start_step:
        raise ValueError("mismatch trust warmup end must be at least its start")
    mismatch_trust_baseline_multiplier = _finite_real(
        mismatch_trust_baseline_multiplier, "mismatch_trust_baseline_multiplier"
    )
    mismatch_trust_ema_decay = _finite_real(mismatch_trust_ema_decay, "mismatch_trust_ema_decay")
    mismatch_trust_dual_lr = _finite_real(mismatch_trust_dual_lr, "mismatch_trust_dual_lr")
    mismatch_trust_dual_beta_max = _finite_real(
        mismatch_trust_dual_beta_max, "mismatch_trust_dual_beta_max"
    )
    mismatch_trust_pg_scale_min = _finite_real(
        mismatch_trust_pg_scale_min, "mismatch_trust_pg_scale_min"
    )
    mismatch_trust_activation_ratio = _finite_real(
        mismatch_trust_activation_ratio, "mismatch_trust_activation_ratio"
    )
    mismatch_trust_pg_activation_ratio = _finite_real(
        mismatch_trust_pg_activation_ratio, "mismatch_trust_pg_activation_ratio"
    )
    mismatch_trust_ess_drop_threshold = _finite_real(
        mismatch_trust_ess_drop_threshold, "mismatch_trust_ess_drop_threshold"
    )
    mismatch_trust_support_drop_threshold = _finite_real(
        mismatch_trust_support_drop_threshold, "mismatch_trust_support_drop_threshold"
    )
    mismatch_trust_length_risk_threshold = _finite_real(
        mismatch_trust_length_risk_threshold, "mismatch_trust_length_risk_threshold"
    )
    if mismatch_trust_baseline_multiplier <= 1.0:
        raise ValueError("mismatch_trust_baseline_multiplier must exceed one")
    if not 0.0 <= mismatch_trust_ema_decay < 1.0:
        raise ValueError("mismatch_trust_ema_decay must be in [0, 1)")
    if mismatch_trust_dual_lr < 0.0 or mismatch_trust_dual_beta_max < 0.0:
        raise ValueError("mismatch trust dual parameters must be non-negative")
    if not 0.0 < mismatch_trust_pg_scale_min <= 1.0:
        raise ValueError("mismatch_trust_pg_scale_min must be in (0, 1]")
    if mismatch_trust_activation_ratio <= 1.0:
        raise ValueError("mismatch_trust_activation_ratio must exceed one")
    if mismatch_trust_pg_activation_ratio <= mismatch_trust_activation_ratio:
        raise ValueError("mismatch_trust_pg_activation_ratio must exceed activation ratio")
    if isinstance(mismatch_trust_persistence_steps, bool) or not isinstance(
        mismatch_trust_persistence_steps, int
    ) or mismatch_trust_persistence_steps <= 0:
        raise ValueError("mismatch_trust_persistence_steps must be a positive integer")
    if min(
        mismatch_trust_ess_drop_threshold,
        mismatch_trust_support_drop_threshold,
        mismatch_trust_length_risk_threshold,
    ) <= 0.0:
        raise ValueError("mismatch trust health thresholds must be positive")
    (
        response_length_cap,
        cap,
        half_life,
        log_ratio_bound,
        historical_sequence_risk_enabled,
        sequence_history_size,
        sequence_min_history,
        sequence_log_scale_floor,
        sequence_risk_saturation_z,
    ) = _validate_config(
        state,
        source_step,
        response_length_cap,
        cap,
        half_life,
        log_ratio_bound,
        historical_sequence_risk_enabled,
        sequence_history_size,
        sequence_min_history,
        sequence_log_scale_floor,
        sequence_risk_saturation_z,
    )
    lower_max_log_bound = _finite_real(lower_max_log_bound, "lower_max_log_bound")
    if not 0.0 < lower_max_log_bound <= MAX_LOG_BOUND:
        raise ValueError("lower_max_log_bound must be positive and at most MAX_LOG_BOUND")
    if adaptive_lower_enabled and lower_max_log_bound < math.log(cap):
        raise ValueError("lower_max_log_bound must be at least log(cap)")
    with torch.no_grad():
        valid, mask_semantics_valid, rows_nonempty = _validate_tensor_inputs(
            actor_old_log_probs, rollout_log_probs, advantages, response_mask
        )
        actor = torch.where(
            valid, actor_old_log_probs.detach().to(dtype=torch.float32), torch.zeros_like(actor_old_log_probs, dtype=torch.float32)
        )
        rollout = torch.where(
            valid, rollout_log_probs.detach().to(dtype=torch.float32), torch.zeros_like(rollout_log_probs, dtype=torch.float32)
        )
        advantage = torch.where(
            valid, advantages.detach().to(dtype=torch.float32), torch.zeros_like(advantages, dtype=torch.float32)
        )
        ones = torch.ones_like(actor, dtype=torch.float32)
        scalar_zero = actor.new_zeros(())
        scalar_one = actor.new_ones(())
        base_inputs_finite = torch.isfinite(actor).all() & torch.isfinite(advantage).all()
        correction_inputs_finite = torch.isfinite(rollout).all()
        controller_inputs_finite = base_inputs_finite & correction_inputs_finite
        safe_actor = torch.nan_to_num(actor, nan=0.0, posinf=0.0, neginf=0.0)
        safe_rollout = torch.nan_to_num(rollout, nan=0.0, posinf=0.0, neginf=0.0)
        safe_advantage = torch.nan_to_num(advantage, nan=0.0, posinf=0.0, neginf=0.0)
        lengths = valid.sum(dim=1).to(torch.float32)
        lengths_within_cap = (lengths <= response_length_cap).all()
        response_length_mean = lengths.mean()
        response_clip_ratio = (lengths >= response_length_cap).to(torch.float32).mean()
        if state.previous_response_length_mean is None:
            growth = scalar_zero
        else:
            growth = torch.clamp(response_length_mean - state.previous_response_length_mean, min=0.0) / response_length_cap
        length_risk = torch.clamp(response_clip_ratio + growth, min=0.0, max=1.0)
        state_alpha = scalar_one * state.alpha
        raw_log_ratio = safe_actor - safe_rollout
        if low_residual_budget_enabled:
            controller_inputs_finite = controller_inputs_finite & torch.isfinite(raw_log_ratio).all()
            raw_log_ratio = torch.nan_to_num(raw_log_ratio)
        log_cap = scalar_one * math.log(cap)
        weight_delta = torch.clamp(raw_log_ratio, min=-log_cap, max=log_cap)
        diagnostic_delta = torch.clamp(raw_log_ratio, min=-log_ratio_bound, max=log_ratio_bound)
        full_lower_bound, full_upper_bound = _fp32_power_bounds(cap, scalar_one)
        full_tis = torch.minimum(torch.maximum(torch.exp(weight_delta), full_lower_bound), full_upper_bound)

        k_terms = torch.clamp_min(torch.expm1(diagnostic_delta) - diagnostic_delta, 0.0)
        k3 = torch.clamp_min(_seq_token_mean(k_terms, valid), 0.0)
        mismatch_severity = -torch.expm1(-torch.sqrt(2.0 * k3))
        mismatch_ema = k3 if state.mismatch_ema is None else (
            mismatch_trust_ema_decay * k3.new_tensor(state.mismatch_ema)
            + (1.0 - mismatch_trust_ema_decay) * k3
        )
        baseline_ema = (
            scalar_zero
            if state.mismatch_baseline_ema is None
            else k3.new_tensor(state.mismatch_baseline_ema)
        )
        baseline_count = state.mismatch_baseline_count
        in_baseline_window = (
            mismatch_trust_region_enabled
            and mismatch_trust_warmup_start_step <= source_step <= mismatch_trust_warmup_end_step
        )
        if in_baseline_window:
            baseline_ema = k3 if baseline_count == 0 else (
                mismatch_trust_ema_decay * baseline_ema
                + (1.0 - mismatch_trust_ema_decay) * k3
            )
            baseline_count += 1
        trust_target = baseline_ema * mismatch_trust_baseline_multiplier
        trust_ready = (
            mismatch_trust_region_enabled
            and source_step > mismatch_trust_warmup_end_step
            and baseline_count > 0
            and float(trust_target.item()) > 0.0
        )
        mismatch_ratio = (
            mismatch_ema / trust_target.clamp_min(_FLOAT32_INFO.tiny) if trust_ready else scalar_zero
        )
        proposed_dual_beta = 0.0
        pg_scale = scalar_one
        sequence_accumulated = torch.where(
            valid, k_terms / float(response_length_cap), torch.zeros_like(k_terms)
        ).sum(dim=1)
        sequence_accumulated_scale = sequence_accumulated.abs().amax()
        safe_sequence_accumulated_scale = torch.where(
            sequence_accumulated_scale > 0.0, sequence_accumulated_scale, scalar_one
        )
        normalized_sequence_accumulated = sequence_accumulated / safe_sequence_accumulated_scale
        sequence_accumulated_mean = normalized_sequence_accumulated.mean() * safe_sequence_accumulated_scale
        sequence_accumulated_rms = (
            torch.sqrt(normalized_sequence_accumulated.square().mean()) * safe_sequence_accumulated_scale
        )
        sequence_accumulated_max = sequence_accumulated.amax()
        sequence_risk_absolute_legacy = torch.clamp(
            torch.sqrt(2.0 * sequence_accumulated_rms), min=0.0, max=1.0
        )
        if historical_sequence_risk_enabled:
            (
                sequence_risk,
                sequence_q_history_ready,
                sequence_q_reference,
                sequence_q_log_median,
                sequence_q_log_scale,
                sequence_q_z_positive,
                sequence_q_severe_incident_raw,
            ) = _historical_sequence_risk(
                sequence_accumulated_rms,
                state.sequence_q_history,
                sequence_min_history,
                sequence_log_scale_floor,
                sequence_risk_saturation_z,
            )
        else:
            sequence_risk = sequence_risk_absolute_legacy
            sequence_q_history_ready = scalar_zero
            sequence_q_reference = scalar_zero
            sequence_q_log_median = scalar_zero
            sequence_q_log_scale = scalar_zero
            sequence_q_z_positive = scalar_zero
            sequence_q_severe_incident_raw = scalar_zero
        historical_sequence_risk_enabled_metric = (
            scalar_one if historical_sequence_risk_enabled else scalar_zero
        )
        sequence_q_history_count = scalar_one * float(len(state.sequence_q_history))
        sequence_q_severe_incident = torch.where(
            controller_inputs_finite, sequence_q_severe_incident_raw, scalar_zero
        )
        if historical_sequence_risk_enabled:
            sequence_q_history_append_proposed_bool = controller_inputs_finite & (
                sequence_q_severe_incident_raw == 0.0
            )
            sequence_q_history_append_proposed = sequence_q_history_append_proposed_bool.to(torch.float32)
            sequence_q_history_proposed_count = torch.where(
                sequence_q_history_append_proposed_bool,
                scalar_one * float(min(len(state.sequence_q_history) + 1, sequence_history_size)),
                sequence_q_history_count,
            )
            sequence_q_history_frozen = sequence_q_severe_incident
        else:
            sequence_q_history_append_proposed = scalar_zero
            sequence_q_history_proposed_count = sequence_q_history_count
            sequence_q_history_frozen = scalar_zero

        advantage_scale = safe_advantage.abs().amax()
        safe_advantage_scale = torch.where(advantage_scale > 0.0, advantage_scale, scalar_one)
        normalized_advantage = safe_advantage / safe_advantage_scale
        safe_lengths = lengths.clamp_min(1.0)
        sequence_advantage_magnitude = torch.where(
            valid,
            safe_advantage.abs() / safe_lengths.unsqueeze(1),
            torch.zeros_like(safe_advantage),
        ).sum(dim=1)
        advantage_max = sequence_advantage_magnitude.amax()
        support_threshold = 16.0 * _FLOAT32_INFO.eps * advantage_max
        advantage_support_fraction = torch.where(
            advantage_max > 0.0,
            (sequence_advantage_magnitude > support_threshold).to(torch.float32).mean(),
            scalar_zero,
        )
        advantage_collapse_risk = length_risk * (1.0 - advantage_support_fraction).square()
        tis_delta = full_tis - 1.0
        numerator = _seq_token_mean(normalized_advantage.square() * tis_delta.square(), valid)
        denominator = _seq_token_mean(normalized_advantage.square(), valid)
        safe_denominator = torch.where(advantage_scale > 0.0, denominator, scalar_one)
        advantage_impact = torch.where(advantage_scale > 0.0, numerator / safe_denominator, scalar_zero)
        advantage_impact_normalized = advantage_impact / (1.0 + advantage_impact)
        contribution_risk = torch.clamp(torch.sqrt(torch.clamp_min(advantage_impact, 0.0)), min=0.0, max=1.0)
        contribution = normalized_advantage * tis_delta
        contribution_mean = _seq_token_mean(contribution, valid)
        contribution_absolute_mean = _seq_token_mean(contribution.abs(), valid)
        safe_contribution_absolute_mean = torch.where(
            contribution_absolute_mean > 0.0, contribution_absolute_mean, scalar_one
        )
        contribution_bias = torch.where(
            contribution_absolute_mean > 0.0,
            contribution_mean.abs() / safe_contribution_absolute_mean,
            scalar_zero,
        )
        alpha_target = torch.maximum(contribution_risk, sequence_risk)
        length_dominant = (length_risk > alpha_target) & length_risk_enabled
        if length_risk_enabled:
            alpha_target = torch.maximum(alpha_target, length_risk)
        contribution_dominant = (contribution_risk >= sequence_risk) & ~length_dominant
        sequence_dominant = (sequence_risk > contribution_risk) & ~length_dominant
        correction_need = alpha_target
        decay = scalar_one * (2.0 ** (-1.0 / half_life))
        released_alpha = decay * state_alpha + (1.0 - decay) * alpha_target
        candidate_alpha = torch.where(alpha_target >= state_alpha, alpha_target, released_alpha)
        controlled_alpha = torch.where(controller_inputs_finite, candidate_alpha, state_alpha)
        applied_alpha = controlled_alpha if apply_control else scalar_zero
        current_lower_bound, current_upper_bound = _fp32_power_bounds(cap, applied_alpha)
        # Use past-only release for this batch; only successful actor updates
        # commit the EMA below. Keep the symmetric full_tis risk reference fixed.
        lower_release = scalar_one * state.lower_release
        proposed_lower_release = lower_release
        lower_log_bound = log_cap
        if adaptive_lower_enabled and apply_control:
            lower_log_bound = log_cap + lower_release * (lower_max_log_bound - log_cap)
            if state.lower_release > 0.0:
                current_lower_bound = _fp32_ceil(
                    torch.exp(-applied_alpha.to(torch.float64) * lower_log_bound.to(torch.float64))
                )
                weight_delta = torch.minimum(torch.maximum(raw_log_ratio, -lower_log_bound), log_cap)
            proposed_lower_release = torch.where(
                controller_inputs_finite,
                decay * lower_release + (1.0 - decay) * alpha_target,
                lower_release,
            )
        candidate_weights = torch.where(
            valid,
            torch.minimum(
                torch.maximum(torch.exp(applied_alpha * weight_delta), current_lower_bound), current_upper_bound
            ),
            ones,
        )
        if weight_mode == "positive_lower_power":
            # Only A > 0 receives the lower floor. A <= 0 retains the upper
            # cap and the same tempering exponent, with no positive floor.
            upper_only_delta = torch.minimum(raw_log_ratio, log_cap)
            # alpha=0 must produce unit weights even if a finite-input
            # subtraction overflowed to -inf in FP32.
            upper_only_delta = torch.where(applied_alpha == 0, scalar_zero, upper_only_delta)
            upper_only_weights = torch.minimum(torch.exp(applied_alpha * upper_only_delta), current_upper_bound)
            candidate_weights = torch.where(
                valid & (safe_advantage <= 0), upper_only_weights, candidate_weights
            )
        if weight_mode == "lower_linear":
            # Interpolate only the low side; retain full upper-only TIS above 1.
            # Clip before exp to avoid overflow even in an unselected branch.
            upper_tis = torch.exp(torch.minimum(raw_log_ratio, log_cap))
            linear_weights = torch.where(
                raw_log_ratio < 0.0,
                (1.0 - applied_alpha) + applied_alpha * torch.exp(torch.minimum(raw_log_ratio, scalar_zero)),
                upper_tis,
            )
            current_lower_bound = 1.0 - applied_alpha
            current_upper_bound = scalar_one * cap
            candidate_weights = torch.where(valid, linear_weights, ones)
        budget_metrics = {}
        if low_residual_budget_enabled:
            candidate_weights, budget_metrics = _apply_low_residual_budget(
                raw_log_ratio, safe_advantage, valid, applied_alpha, candidate_weights, low_residual_budget_beta
            )
            # There is no fixed positive floor in this mode.
            current_lower_bound = scalar_zero
        controlled_weights = torch.where(controller_inputs_finite, candidate_weights, ones)
        weights = (controlled_weights if apply_control else ones).detach()
        proposed_alpha = applied_alpha

        low_ratio = valid & (raw_log_ratio < 0.0)
        low_ratio_fraction = _seq_token_mean(low_ratio.to(torch.float32), valid)
        low_ratio_lift = torch.where(low_ratio, weights - full_tis, torch.zeros_like(weights))
        low_ratio_weight_lift_mean = _conditional_weight_mean(low_ratio_lift + 1.0, low_ratio) - 1.0
        valid_any = valid.any()
        weight_min = torch.where(valid_any, weights.masked_fill(~valid, float("inf")).amin(), scalar_one)
        weight_max = torch.where(valid_any, weights.masked_fill(~valid, float("-inf")).amax(), scalar_one)
        valid_count = valid.sum().to(torch.float32).clamp_min(1.0)
        weight_scale = weights.masked_fill(~valid, 0.0).amax().clamp_min(_FLOAT32_INFO.tiny)
        normalized_weights = torch.where(valid, weights / weight_scale, torch.zeros_like(weights))
        normalized_weight_sum = normalized_weights.sum()
        normalized_weight_square_sum = normalized_weights.square().sum()
        effective_sample_size = normalized_weight_sum.square() / (
            valid_count * normalized_weight_square_sum.clamp_min(_FLOAT32_INFO.tiny)
        )
        ess_baseline = (
            scalar_zero if state.mismatch_ess_baseline_ema is None
            else k3.new_tensor(state.mismatch_ess_baseline_ema)
        )
        support_baseline = (
            scalar_zero if state.mismatch_support_baseline_ema is None
            else k3.new_tensor(state.mismatch_support_baseline_ema)
        )
        if in_baseline_window:
            ess_baseline = effective_sample_size if state.mismatch_ess_baseline_ema is None else (
                mismatch_trust_ema_decay * ess_baseline
                + (1.0 - mismatch_trust_ema_decay) * effective_sample_size
            )
            support_baseline = advantage_support_fraction if state.mismatch_support_baseline_ema is None else (
                mismatch_trust_ema_decay * support_baseline
                + (1.0 - mismatch_trust_ema_decay) * advantage_support_fraction
            )
        excess_steps = (
            state.mismatch_excess_steps + 1
            if trust_ready and float(mismatch_ratio.item()) > mismatch_trust_activation_ratio
            else 0
        )
        persistent_excess = trust_ready and excess_steps >= mismatch_trust_persistence_steps
        ess_drop_score = torch.clamp(
            (ess_baseline - effective_sample_size) / mismatch_trust_ess_drop_threshold,
            min=0.0,
            max=1.0,
        )
        support_drop_score = torch.clamp(
            (support_baseline - advantage_support_fraction) / mismatch_trust_support_drop_threshold,
            min=0.0,
            max=1.0,
        )
        length_harm_score = torch.clamp(
            (length_risk - mismatch_trust_length_risk_threshold)
            / mismatch_trust_length_risk_threshold,
            min=0.0,
            max=1.0,
        )
        structural_harm_score = sequence_risk * torch.maximum(ess_drop_score, support_drop_score)
        harm_score = torch.maximum(structural_harm_score, length_harm_score)
        control_gate = persistent_excess and float(harm_score.item()) > 0.0
        if trust_ready:
            if control_gate:
                pressure = float(harm_score.item()) * max(
                    0.0, float(mismatch_ratio.item()) - mismatch_trust_activation_ratio
                )
                proposed_dual_beta = min(
                    mismatch_trust_dual_beta_max,
                    max(0.0, state.mismatch_dual_beta + mismatch_trust_dual_lr * pressure),
                )
            else:
                recovery = max(0.0, 1.0 - float(mismatch_ratio.item()))
                proposed_dual_beta = max(0.0, state.mismatch_dual_beta - mismatch_trust_dual_lr * recovery)
            if control_gate and float(mismatch_ratio.item()) > mismatch_trust_pg_activation_ratio:
                raw_pg_scale = torch.clamp(
                    mismatch_trust_pg_activation_ratio / mismatch_ratio,
                    min=mismatch_trust_pg_scale_min,
                    max=1.0,
                )
                pg_scale = 1.0 - harm_score * (1.0 - raw_pg_scale)
        controlled_weight_clip_fraction = _seq_token_mean(
            (
                valid
                & (((weights <= current_lower_bound) & ((safe_advantage > 0) if weight_mode == "positive_lower_power" else valid))
                   | (weights >= current_upper_bound))
            ).to(torch.float32),
            valid,
        )
        weight_clip_fraction = controlled_weight_clip_fraction if apply_control else scalar_zero
        metrics = {
            "enabled": scalar_one if apply_control else scalar_zero,
            "telemetry_enabled": scalar_one,
            "sequence_risk_probe_only": scalar_zero if apply_control else scalar_one,
            "weight_mode_symmetric_power": scalar_one if weight_mode == "symmetric_power" else scalar_zero,
            "weight_mode_lower_linear": scalar_one if weight_mode == "lower_linear" else scalar_zero,
            "alpha_previous": state_alpha,
            "alpha_applied": applied_alpha,
            "alpha_proposed": proposed_alpha,
            "alpha_target": torch.where(controller_inputs_finite, alpha_target, scalar_zero),
            "effective_cap": current_upper_bound,
            "effective_lower_bound": scalar_zero if weight_mode == "positive_lower_power" and apply_control else current_lower_bound,
            "positive_effective_lower_bound": current_lower_bound,
            "weight_mode_positive_lower_power": scalar_one if weight_mode == "positive_lower_power" else scalar_zero,
            "adaptive_lower_enabled": scalar_one if adaptive_lower_enabled else scalar_zero,
            "lower_log_bound": lower_log_bound,
            "lower_release_applied": lower_release if adaptive_lower_enabled and apply_control else scalar_zero,
            "lower_release_proposed": proposed_lower_release,
            "k3": torch.where(controller_inputs_finite, k3, scalar_zero),
            "mismatch_severity": torch.where(controller_inputs_finite, mismatch_severity, scalar_zero),
            "mismatch_trust_enabled": scalar_one if mismatch_trust_region_enabled else scalar_zero,
            "mismatch_trust_ema": torch.where(controller_inputs_finite, mismatch_ema, scalar_zero),
            "mismatch_trust_baseline_ema": baseline_ema,
            "mismatch_trust_baseline_count": scalar_one * float(baseline_count),
            "mismatch_trust_target": trust_target,
            "mismatch_trust_ready": scalar_one if trust_ready else scalar_zero,
            "mismatch_trust_ratio": torch.where(controller_inputs_finite, mismatch_ratio, scalar_zero),
            "mismatch_trust_dual_beta": scalar_one * proposed_dual_beta,
            "mismatch_trust_pg_scale": pg_scale,
            "mismatch_trust_activation_ratio": scalar_one * mismatch_trust_activation_ratio,
            "mismatch_trust_pg_activation_ratio": scalar_one * mismatch_trust_pg_activation_ratio,
            "mismatch_trust_excess_steps": scalar_one * float(excess_steps),
            "mismatch_trust_persistent_excess": scalar_one if persistent_excess else scalar_zero,
            "mismatch_trust_ess_baseline_ema": ess_baseline,
            "mismatch_trust_support_baseline_ema": support_baseline,
            "mismatch_trust_ess_drop_score": ess_drop_score,
            "mismatch_trust_support_drop_score": support_drop_score,
            "mismatch_trust_length_harm_score": length_harm_score,
            "mismatch_trust_structural_harm_score": structural_harm_score,
            "mismatch_trust_harm_score": harm_score,
            "mismatch_trust_control_gate": scalar_one if control_gate else scalar_zero,
            "advantage_support_fraction": torch.where(
                controller_inputs_finite, advantage_support_fraction, scalar_zero
            ),
            "advantage_collapse_risk": torch.where(controller_inputs_finite, advantage_collapse_risk, scalar_zero),
            "advantage_impact": torch.where(controller_inputs_finite, advantage_impact, scalar_zero),
            "advantage_impact_normalized": torch.where(controller_inputs_finite, advantage_impact_normalized, scalar_zero),
            "contribution_risk": torch.where(controller_inputs_finite, contribution_risk, scalar_zero),
            "contribution_bias": torch.where(controller_inputs_finite, contribution_bias, scalar_zero),
            "sequence_accumulated_k3_mean": torch.where(
                controller_inputs_finite, sequence_accumulated_mean, scalar_zero
            ),
            "sequence_accumulated_k3_rms": torch.where(
                controller_inputs_finite, sequence_accumulated_rms, scalar_zero
            ),
            "sequence_accumulated_k3_max": torch.where(
                controller_inputs_finite, sequence_accumulated_max, scalar_zero
            ),
            "sequence_risk": torch.where(controller_inputs_finite, sequence_risk, scalar_zero),
            "sequence_risk_absolute_legacy": torch.where(
                controller_inputs_finite, sequence_risk_absolute_legacy, scalar_zero
            ),
            "historical_sequence_risk_enabled": historical_sequence_risk_enabled_metric,
            "sequence_q_history_count": sequence_q_history_count,
            "sequence_q_history_proposed_count": sequence_q_history_proposed_count,
            "sequence_q_history_ready": sequence_q_history_ready,
            "sequence_q_reference": sequence_q_reference,
            "sequence_q_log_median": sequence_q_log_median,
            "sequence_q_log_scale": sequence_q_log_scale,
            "sequence_q_z_positive": torch.where(
                controller_inputs_finite, sequence_q_z_positive, scalar_zero
            ),
            "sequence_q_history_append_proposed": sequence_q_history_append_proposed,
            "sequence_q_severe_incident": sequence_q_severe_incident,
            "sequence_q_history_frozen": sequence_q_history_frozen,
            "contribution_dominant": torch.where(
                controller_inputs_finite, contribution_dominant.to(torch.float32), scalar_zero
            ),
            "sequence_dominant": torch.where(
                controller_inputs_finite, sequence_dominant.to(torch.float32), scalar_zero
            ),
            "correction_need": torch.where(controller_inputs_finite, correction_need, scalar_zero),
            "response_clip_ratio": response_clip_ratio,
            "response_length_mean": response_length_mean,
            "response_length_growth_ratio": growth,
            "length_risk": length_risk,
            "length_risk_enabled": scalar_one if length_risk_enabled else scalar_zero,
            "length_dominant": torch.where(
                controller_inputs_finite, length_dominant.to(torch.float32), scalar_zero
            ),
            "risk": torch.where(controller_inputs_finite, alpha_target, scalar_zero),
            "weight_min": weight_min,
            "weight_mean": _seq_token_mean(weights, valid),
            "weight_max": weight_max,
            "effective_sample_size": effective_sample_size,
            "weight_clip_fraction": weight_clip_fraction,
            "low_ratio_fraction": torch.where(controller_inputs_finite, low_ratio_fraction, scalar_zero),
            "low_ratio_weight_lift_mean": torch.where(controller_inputs_finite, low_ratio_weight_lift_mean, scalar_zero),
            "positive_advantage_weight_mean": torch.where(
                controller_inputs_finite, _conditional_weight_mean(weights, valid & (safe_advantage > 0.0)), scalar_one
            ),
            "negative_advantage_weight_mean": torch.where(
                controller_inputs_finite, _conditional_weight_mean(weights, valid & (safe_advantage < 0.0)), scalar_one
            ),
            "base_inputs_finite": base_inputs_finite.to(torch.float32),
            "correction_inputs_finite": correction_inputs_finite.to(torch.float32),
            "state_held": (~controller_inputs_finite).to(torch.float32),
            "actor_update_allowed": (
                (controller_inputs_finite if low_residual_budget_enabled else base_inputs_finite).to(torch.float32)
                if apply_control else scalar_one
            ),
            "source_step": scalar_one * float(source_step),
        }
        metrics.update({"low_residual_" + key: value for key, value in budget_metrics.items()})
        metric_names = tuple(metrics)
        telemetry = torch.stack(tuple(metrics[name] for name in metric_names) + (
            mask_semantics_valid.to(torch.float32),
            rows_nonempty.to(torch.float32),
            lengths_within_cap.to(torch.float32),
        ))
        telemetry_values = telemetry.detach().to(device="cpu").numpy().tolist()
        metric_values = dict(zip(metric_names, telemetry_values[: len(metric_names)]))
        mask_valid_value, rows_nonempty_value, lengths_within_cap_value = telemetry_values[len(metric_names) :]
        if mask_valid_value == 0.0:
            raise ValueError("response_mask must be boolean or contain only finite 0/1 values")
        if rows_nonempty_value == 0.0:
            raise ValueError("every response row must contain at least one valid token")
        if lengths_within_cap_value == 0.0:
            raise ValueError("observed response length must not exceed response_length_cap")
        state_held = metric_values["state_held"] != 0.0
        proposed_sequence_q_history = state.sequence_q_history
        if not state_held and metric_values["sequence_q_history_append_proposed"] != 0.0:
            proposed_sequence_q_history = (
                *state.sequence_q_history,
                float(metric_values["sequence_accumulated_k3_rms"]),
            )[-sequence_history_size:]
        hold_controller_state = state_held and apply_control
        proposed_state = state if hold_controller_state else ExplorationTISState(
            alpha=min(max(float(metric_values["alpha_proposed"]), 0.0), 1.0),
            previous_response_length_mean=float(metric_values["response_length_mean"]),
            source_step=source_step,
            sequence_q_history=proposed_sequence_q_history,
            lower_release=(
                min(max(float(metric_values["lower_release_proposed"]), 0.0), 1.0)
                if adaptive_lower_enabled and apply_control else state.lower_release
            ),
            mismatch_ema=float(metric_values["mismatch_trust_ema"]),
            mismatch_baseline_ema=(
                float(metric_values["mismatch_trust_baseline_ema"])
                if int(metric_values["mismatch_trust_baseline_count"]) > 0 else None
            ),
            mismatch_baseline_count=int(metric_values["mismatch_trust_baseline_count"]),
            mismatch_dual_beta=float(metric_values["mismatch_trust_dual_beta"]),
            mismatch_ess_baseline_ema=(
                float(metric_values["mismatch_trust_ess_baseline_ema"])
                if int(metric_values["mismatch_trust_baseline_count"]) > 0 else None
            ),
            mismatch_support_baseline_ema=(
                float(metric_values["mismatch_trust_support_baseline_ema"])
                if int(metric_values["mismatch_trust_baseline_count"]) > 0 else None
            ),
            mismatch_excess_steps=int(metric_values["mismatch_trust_excess_steps"]),
        )
        return ExplorationTISResult(
            weights=weights,
            proposed_state=proposed_state,
            actor_update_allowed=metric_values["actor_update_allowed"] != 0.0,
            metrics={
                **{(_METRIC_PREFIX + key if not key.startswith("low_residual_") else "rollout_corr/" + key):
                   float(value) for key, value in metric_values.items()},
                **({"rollout_corr/low_residual_invalid_batch": float(state_held)} if low_residual_budget_enabled else {}),
            },
        )
