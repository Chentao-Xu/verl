# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Low-precision quantize-dequantize simulation for NPU vLLM rollouts.

This module intentionally keeps vLLM weights and GEMMs in BF16. The external
``quant_cy_npu`` package injects configurable rounding into source weights and,
optionally, Linear activations. It is an accuracy simulator, not a native low-
precision inference backend.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)

_ENABLED_ENV = "VERL_VLLM_MXFP8_SIM_ENABLED"
_FORMAT_ENV = "VERL_LOW_PRECISION_SIM_FORMAT"
_WEIGHT_FORMAT_ENV = "VERL_LOW_PRECISION_WEIGHT_FORMAT"
_ACTIVATION_FORMAT_ENV = "VERL_LOW_PRECISION_ACTIVATION_FORMAT"
_MODE_ENV = "VERL_MXFP8_SIM_MODE"
_SCOPE_ENV = "VERL_MXFP8_SIM_SCOPE"
_LANGUAGE_LAYER_RANGE_ENV = "VERL_MXFP8_SIM_LANGUAGE_LAYER_RANGE"
_ACTIVATION_HOOK_ENV = "VERL_MXFP8_SIM_ACTIVATION_HOOK"
_ACTIVATION_GROUPS_ENV = "VERL_MXFP8_SIM_ACTIVATION_GROUPS"
_VALID_MODES = {"weight_only", "w8a8"}
_VALID_FORMATS = {"mxfp4", "mxfp8", "hif4", "hif8"}
_VALID_SCOPES = {"all", "language_only", "visual_only"}
_VALID_ACTIVATION_HOOKS = {"apply", "forward"}
_VALID_ACTIVATION_GROUPS = {"qkv", "o", "gate_up", "down", "other"}
_ACTIVATION_MARKER = "_verl_mxfp8_simulate_activation"
_ACTIVATION_NAME = "_verl_mxfp8_sim_module_name"
_ROLLOUT_OPERAND_FORMATS = {
    "mxfp8_sim": ("mxfp8", "mxfp8"),
    "hif8_sim": ("hif8", "hif8"),
    "mxfp4_w4a8_sim": ("mxfp4", "mxfp8"),
    "hif4_w4a8_sim": ("hif4", "hif8"),
    "hif4_w4a4_sim": ("hif4", "hif4"),
}


@dataclass
class MXFP8SimStats:
    weight_tensors: int = 0
    weight_elements: int = 0
    activation_calls: int = 0
    activation_elements: int = 0
    activation_calls_by_group: dict[str, int] | None = None


_stats = MXFP8SimStats()
_linear_patch_applied = False
_original_unquantized_linear_apply = None
_original_default_loader_get_all_weights = None
_original_default_loader_load_weights = None
_source_name_audit_groups: set[str] = set()


def is_mxfp8_sim_enabled() -> bool:
    return os.environ.get(_ENABLED_ENV, "0") == "1"


def get_low_precision_sim_format() -> str:
    value = os.environ.get(_FORMAT_ENV, "mxfp8").lower()
    if value not in _VALID_FORMATS:
        raise ValueError(f"{_FORMAT_ENV} must be one of {sorted(_VALID_FORMATS)}, got {value!r}")
    return value


def _get_operand_format(env_name: str) -> str:
    value = os.environ.get(env_name, get_low_precision_sim_format()).lower()
    if value not in _VALID_FORMATS:
        raise ValueError(f"{env_name} must be one of {sorted(_VALID_FORMATS)}, got {value!r}")
    return value


def get_low_precision_weight_format() -> str:
    """Return the simulated format for Linear weights."""
    return _get_operand_format(_WEIGHT_FORMAT_ENV)


def get_low_precision_activation_format() -> str:
    """Return the simulated format for Linear activations."""
    return _get_operand_format(_ACTIVATION_FORMAT_ENV)


def get_rollout_operand_formats(quantization: str) -> tuple[str, str]:
    """Map a public rollout quantization name to weight/activation formats."""
    try:
        return _ROLLOUT_OPERAND_FORMATS[quantization]
    except KeyError as exc:
        raise ValueError(f"Unsupported numerical rollout simulation: {quantization!r}") from exc


def get_mxfp8_sim_mode() -> str:
    mode = os.environ.get(_MODE_ENV, "w8a8").lower()
    if mode not in _VALID_MODES:
        raise ValueError(f"{_MODE_ENV} must be one of {sorted(_VALID_MODES)}, got {mode!r}")
    return mode


def get_mxfp8_sim_scope() -> str:
    scope = os.environ.get(_SCOPE_ENV, "all").lower()
    if scope not in _VALID_SCOPES:
        raise ValueError(f"{_SCOPE_ENV} must be one of {sorted(_VALID_SCOPES)}, got {scope!r}")
    return scope


def get_mxfp8_activation_hook() -> str:
    hook = os.environ.get(_ACTIVATION_HOOK_ENV, "apply").lower()
    if hook not in _VALID_ACTIVATION_HOOKS:
        raise ValueError(f"{_ACTIVATION_HOOK_ENV} must be one of {sorted(_VALID_ACTIVATION_HOOKS)}, got {hook!r}")
    return hook


def get_mxfp8_activation_groups() -> set[str]:
    value = os.environ.get(_ACTIVATION_GROUPS_ENV, "all").strip().lower()
    if value in {"", "all"}:
        return set(_VALID_ACTIVATION_GROUPS)
    groups = {item.strip() for item in value.split(",") if item.strip()}
    invalid = groups - _VALID_ACTIVATION_GROUPS
    if invalid:
        raise ValueError(
            f"{_ACTIVATION_GROUPS_ENV} contains unsupported groups {sorted(invalid)}; "
            f"valid groups are {sorted(_VALID_ACTIVATION_GROUPS)}"
        )
    return groups


def classify_mxfp8_activation_module(name: str) -> str:
    lowered = name.lower()
    leaf = lowered.rsplit(".", maxsplit=1)[-1]
    if leaf in {"qkv_proj", "q_proj", "k_proj", "v_proj"}:
        return "qkv"
    if leaf in {"o_proj", "out_proj"}:
        return "o"
    if leaf in {"gate_up_proj", "gate_proj", "up_proj"}:
        return "gate_up"
    if leaf == "down_proj":
        return "down"
    return "other"


def should_quantize_activation_module(name: str) -> bool:
    return _in_simulation_scope(name) and classify_mxfp8_activation_module(name) in get_mxfp8_activation_groups()


def _record_activation_qdq(name: str, tensor: torch.Tensor) -> None:
    group = classify_mxfp8_activation_module(name)
    if _stats.activation_calls_by_group is None:
        _stats.activation_calls_by_group = {}
    _stats.activation_calls += 1
    _stats.activation_elements += tensor.numel()
    _stats.activation_calls_by_group[group] = _stats.activation_calls_by_group.get(group, 0) + 1


def _is_visual_name(name: str) -> bool:
    lowered = name.lower()
    return lowered.startswith("visual.") or ".visual." in lowered


def _in_language_layer_range(name: str) -> bool:
    value = os.environ.get(_LANGUAGE_LAYER_RANGE_ENV, "").strip()
    if not value:
        return True

    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name.lower())
    if match is None:
        return False
    try:
        start_text, end_text = value.split(":", maxsplit=1)
        start, end = int(start_text), int(end_text)
    except ValueError as exc:
        raise ValueError(f"{_LANGUAGE_LAYER_RANGE_ENV} must be START:END, got {value!r}") from exc
    if start < 0 or end < start:
        raise ValueError(f"{_LANGUAGE_LAYER_RANGE_ENV} must satisfy 0 <= START <= END, got {value!r}")
    return start <= int(match.group(1)) <= end


def _in_simulation_scope(name: str) -> bool:
    scope = get_mxfp8_sim_scope()
    is_visual = _is_visual_name(name)
    if scope == "language_only":
        return not is_visual and _in_language_layer_range(name)
    if scope == "visual_only":
        return is_visual
    return is_visual or _in_language_layer_range(name)


def should_quantize_source_weight(name: str, tensor: torch.Tensor) -> bool:
    """Select logical Linear weights before vLLM packing/sharding."""
    if not name.endswith(".weight") or tensor.ndim != 2:
        return False

    if not _in_simulation_scope(name):
        return False

    lowered = name.lower()
    excluded = (
        "embed_tokens",
        "lm_head",
        "layernorm",
        "layer_norm",
        ".norm",
        "patch_embed",
    )
    return not any(token in lowered for token in excluded)


def _qtype_name_for_format(sim_format: str) -> str:
    if sim_format == "mxfp4":
        return "mxfp4"
    if sim_format == "hif4":
        return "hifx4"
    if sim_format == "mxfp8":
        return "mxfp8e4m3"
    if sim_format == "hif8":
        return "dhif8"
    raise ValueError(f"Unsupported low-precision simulation format: {sim_format!r}")


def _quant_dequant_last_dim(tensor: torch.Tensor, sim_format: str) -> torch.Tensor:
    try:
        from quant_cy_npu.QTensor import delay_quant_dequant_float, quant_dequant_float
        from quant_cy_npu.QType import QType
    except ImportError as exc:
        raise RuntimeError(
            "Low-precision simulation requires quant_cy_npu on PYTHONPATH and its cusrc directory on LD_LIBRARY_PATH"
        ) from exc

    def qdq(value: torch.Tensor) -> torch.Tensor:
        if sim_format == "hif8":
            amax = float(value.detach().abs().max())
            return delay_quant_dequant_float(
                value,
                amax,
                15,
                QType("dhif8").dim_(-1),
                force_py=False,
            )
        return quant_dequant_float(value, QType(_qtype_name_for_format(sim_format)).dim_(-1), force_py=False)

    if tensor.device.type == "npu":
        return qdq(tensor)

    # DefaultModelLoader streams checkpoint tensors from CPU. Stage one tensor
    # at a time through the active NPU because quant_cy_npu has no CPU kernel.
    staged = tensor.to("npu")
    quantized = qdq(staged)
    return quantized.to(tensor.device)


@torch.no_grad()
def quant_dequant_weight_updates(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterator[tuple[str, torch.Tensor]]:
    """Apply configured low-precision QDQ to logical Linear source weights lazily."""
    global _source_name_audit_groups
    before_tensors = _stats.weight_tensors
    before_elements = _stats.weight_elements
    source_name_examples = []
    try:
        for name, tensor in weights:
            if len(source_name_examples) < 8 and name.endswith(".weight") and tensor.ndim == 2:
                source_name_examples.append(name)
            if should_quantize_source_weight(name, tensor):
                tensor = _quant_dequant_last_dim(tensor, get_low_precision_weight_format())
                _stats.weight_tensors += 1
                _stats.weight_elements += tensor.numel()
            yield name, tensor
    finally:
        quantized_tensors = _stats.weight_tensors - before_tensors
        # Dynamic weight sync receives views into a sender-owned, reusable IPC
        # bucket. quant_dequant_float launches asynchronously, so the sender
        # could overwrite that bucket before the NPU has consumed its input.
        # Finish both QDQ and the downstream vLLM copy before returning from
        # this bucket's model.load_weights call.
        if quantized_tensors:
            torch.npu.synchronize()
        if source_name_examples:
            audit_group = "visual" if _is_visual_name(source_name_examples[0]) else "non_visual"
            if audit_group not in _source_name_audit_groups:
                logger.warning(
                    "%s %s source weight name examples: %s",
                    get_low_precision_weight_format(),
                    audit_group,
                    source_name_examples,
                )
                _source_name_audit_groups.add(audit_group)
        logger.warning(
            "%s simulation quantized %d weight tensors (%d elements) for this reload",
            get_low_precision_weight_format(),
            quantized_tensors,
            _stats.weight_elements - before_elements,
        )


def mark_mxfp8_sim_linears(model: torch.nn.Module) -> int:
    """Mark vLLM Linear layers whose activations should receive QDQ."""
    if get_mxfp8_sim_mode() != "w8a8":
        return 0

    try:
        from vllm.model_executor.layers.linear import LinearBase
    except ImportError as exc:
        raise RuntimeError("Could not import vLLM LinearBase for low-precision simulation") from exc

    marked = 0
    marked_by_group: dict[str, int] = {}
    for name, module in model.named_modules():
        if isinstance(module, LinearBase) and "lm_head" not in name.lower() and should_quantize_activation_module(name):
            setattr(module, _ACTIVATION_MARKER, True)
            setattr(module, _ACTIVATION_NAME, name)
            group = classify_mxfp8_activation_module(name)
            marked_by_group[group] = marked_by_group.get(group, 0) + 1
            marked += 1
    logger.warning(
        "Low-precision simulation marked %d Linear layers for activation QDQ "
        "(format=%s, scope=%s, hook=%s, groups=%s, by_group=%s)",
        marked,
        get_low_precision_activation_format(),
        get_mxfp8_sim_scope(),
        get_mxfp8_activation_hook(),
        sorted(get_mxfp8_activation_groups()),
        marked_by_group,
    )
    return marked


def patch_mxfp8_sim_linear_forwards(model: torch.nn.Module) -> int:
    """Wrap marked LinearBase.forward methods for the Omni-style hook experiment."""
    if get_mxfp8_sim_mode() != "w8a8" or get_mxfp8_activation_hook() != "forward":
        return 0

    patched = 0
    for _, module in model.named_modules():
        if not getattr(module, _ACTIVATION_MARKER, False) or getattr(module, "_verl_mxfp8_forward_patched", False):
            continue
        original_forward = module.forward
        module_name = getattr(module, _ACTIVATION_NAME, "")

        def forward(x, *args, __original_forward=original_forward, __module_name=module_name, **kwargs):
            x = _quant_dequant_last_dim(x, get_low_precision_activation_format())
            _record_activation_qdq(__module_name, x)
            return __original_forward(x, *args, **kwargs)

        module.forward = forward
        setattr(module, "_verl_mxfp8_forward_patched", True)
        patched += 1
    logger.warning(
        "Low-precision simulation wrapped %d LinearBase.forward methods (activation=%s)",
        patched,
        get_low_precision_activation_format(),
    )
    return patched


def apply_mxfp8_sim_patches() -> None:
    """Patch vLLM checkpoint loading and optional Linear activation QDQ."""
    global _linear_patch_applied, _original_default_loader_get_all_weights
    global _original_default_loader_load_weights, _original_unquantized_linear_apply
    if _linear_patch_applied:
        return

    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    original_apply = UnquantizedLinearMethod.apply

    def _patched_apply(self, layer, x, bias=None):
        if get_mxfp8_activation_hook() == "apply" and getattr(layer, _ACTIVATION_MARKER, False):
            x = _quant_dequant_last_dim(x, get_low_precision_activation_format())
            _record_activation_qdq(getattr(layer, _ACTIVATION_NAME, ""), x)
        return original_apply(self, layer, x, bias)

    _original_unquantized_linear_apply = original_apply
    UnquantizedLinearMethod.apply = _patched_apply

    original_get_all_weights = DefaultModelLoader.get_all_weights

    def _patched_get_all_weights(self, model_config, model):
        weights = original_get_all_weights(self, model_config, model)
        return quant_dequant_weight_updates(weights)

    _original_default_loader_get_all_weights = original_get_all_weights
    DefaultModelLoader.get_all_weights = _patched_get_all_weights

    original_load_weights = DefaultModelLoader.load_weights

    def _patched_load_weights(self, model, model_config):
        result = original_load_weights(self, model, model_config)
        mark_mxfp8_sim_linears(model)
        patch_mxfp8_sim_linear_forwards(model)
        return result

    _original_default_loader_load_weights = original_load_weights
    DefaultModelLoader.load_weights = _patched_load_weights
    _linear_patch_applied = True
    logger.warning(
        "Applied low-precision simulation patch (weight=%s, activation=%s, mode=%s, scope=%s, "
        "language_layer_range=%s, activation_hook=%s, activation_groups=%s, gemm=bf16)",
        get_low_precision_weight_format(),
        get_low_precision_activation_format(),
        get_mxfp8_sim_mode(),
        get_mxfp8_sim_scope(),
        os.environ.get(_LANGUAGE_LAYER_RANGE_ENV, ""),
        get_mxfp8_activation_hook(),
        sorted(get_mxfp8_activation_groups()),
    )


def get_mxfp8_sim_stats() -> MXFP8SimStats:
    return MXFP8SimStats(**vars(_stats))


__all__ = [
    "apply_mxfp8_sim_patches",
    "get_mxfp8_sim_mode",
    "get_low_precision_activation_format",
    "get_low_precision_sim_format",
    "get_low_precision_weight_format",
    "get_rollout_operand_formats",
    "get_mxfp8_sim_scope",
    "get_mxfp8_activation_groups",
    "get_mxfp8_activation_hook",
    "get_mxfp8_sim_stats",
    "is_mxfp8_sim_enabled",
    "mark_mxfp8_sim_linears",
    "patch_mxfp8_sim_linear_forwards",
    "quant_dequant_weight_updates",
    "should_quantize_source_weight",
    "should_quantize_activation_module",
]
