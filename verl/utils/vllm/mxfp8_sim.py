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

"""MXFP8 quantize-dequantize simulation for NPU vLLM rollouts.

This module intentionally keeps vLLM weights and GEMMs in BF16. The external
``quant_cy_npu`` package is used to inject MXFP8 rounding into source weights
and, optionally, Linear activations. It is an accuracy simulator, not a low
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
_MODE_ENV = "VERL_MXFP8_SIM_MODE"
_SCOPE_ENV = "VERL_MXFP8_SIM_SCOPE"
_LANGUAGE_LAYER_RANGE_ENV = "VERL_MXFP8_SIM_LANGUAGE_LAYER_RANGE"
_VALID_MODES = {"weight_only", "w8a8"}
_VALID_SCOPES = {"all", "language_only", "visual_only"}
_ACTIVATION_MARKER = "_verl_mxfp8_simulate_activation"


@dataclass
class MXFP8SimStats:
    weight_tensors: int = 0
    weight_elements: int = 0
    activation_calls: int = 0
    activation_elements: int = 0


_stats = MXFP8SimStats()
_linear_patch_applied = False
_original_unquantized_linear_apply = None
_source_name_audit_groups: set[str] = set()


def is_mxfp8_sim_enabled() -> bool:
    return os.environ.get(_ENABLED_ENV, "0") == "1"


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


def _quant_dequant_last_dim(tensor: torch.Tensor) -> torch.Tensor:
    try:
        from quant_cy_npu.QTensor import quant_dequant_float
        from quant_cy_npu.QType import QType
    except ImportError as exc:
        raise RuntimeError(
            "MXFP8 simulation requires quant_cy_npu on PYTHONPATH and its cusrc directory on LD_LIBRARY_PATH"
        ) from exc

    qtype = QType("mxfp8e4m3").dim_(-1)
    return quant_dequant_float(tensor, qtype, force_py=False)


@torch.no_grad()
def quant_dequant_weight_updates(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterator[tuple[str, torch.Tensor]]:
    """Apply MXFP8 QDQ to logical Linear source weights lazily."""
    global _source_name_audit_groups
    before_tensors = _stats.weight_tensors
    before_elements = _stats.weight_elements
    source_name_examples = []
    try:
        for name, tensor in weights:
            if len(source_name_examples) < 8 and name.endswith(".weight") and tensor.ndim == 2:
                source_name_examples.append(name)
            if should_quantize_source_weight(name, tensor):
                tensor = _quant_dequant_last_dim(tensor)
                _stats.weight_tensors += 1
                _stats.weight_elements += tensor.numel()
            yield name, tensor
    finally:
        if source_name_examples:
            audit_group = "visual" if _is_visual_name(source_name_examples[0]) else "non_visual"
            if audit_group not in _source_name_audit_groups:
                logger.warning("MXFP8 %s source weight name examples: %s", audit_group, source_name_examples)
                _source_name_audit_groups.add(audit_group)
        logger.warning(
            "MXFP8 simulation quantized %d weight tensors (%d elements) for this reload",
            _stats.weight_tensors - before_tensors,
            _stats.weight_elements - before_elements,
        )


def mark_mxfp8_sim_linears(model: torch.nn.Module) -> int:
    """Mark vLLM Linear layers whose activations should receive QDQ."""
    if get_mxfp8_sim_mode() != "w8a8":
        return 0

    try:
        from vllm.model_executor.layers.linear import LinearBase
    except ImportError as exc:
        raise RuntimeError("Could not import vLLM LinearBase for MXFP8 simulation") from exc

    marked = 0
    for name, module in model.named_modules():
        if isinstance(module, LinearBase) and "lm_head" not in name.lower() and _in_simulation_scope(name):
            setattr(module, _ACTIVATION_MARKER, True)
            marked += 1
    logger.warning(
        "MXFP8 simulation marked %d Linear layers for activation QDQ (scope=%s)",
        marked,
        get_mxfp8_sim_scope(),
    )
    return marked


def apply_mxfp8_sim_patches() -> None:
    """Patch unquantized vLLM Linear apply with optional activation QDQ."""
    global _linear_patch_applied, _original_unquantized_linear_apply
    if _linear_patch_applied:
        return

    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    original_apply = UnquantizedLinearMethod.apply

    def _patched_apply(self, layer, x, bias=None):
        if getattr(layer, _ACTIVATION_MARKER, False):
            x = _quant_dequant_last_dim(x)
            _stats.activation_calls += 1
            _stats.activation_elements += x.numel()
        return original_apply(self, layer, x, bias)

    _original_unquantized_linear_apply = original_apply
    UnquantizedLinearMethod.apply = _patched_apply
    _linear_patch_applied = True
    logger.warning(
        "Applied MXFP8 simulation patch (mode=%s, scope=%s)",
        get_mxfp8_sim_mode(),
        get_mxfp8_sim_scope(),
    )


def get_mxfp8_sim_stats() -> MXFP8SimStats:
    return MXFP8SimStats(**vars(_stats))


__all__ = [
    "apply_mxfp8_sim_patches",
    "get_mxfp8_sim_mode",
    "get_mxfp8_sim_scope",
    "get_mxfp8_sim_stats",
    "is_mxfp8_sim_enabled",
    "mark_mxfp8_sim_linears",
    "quant_dequant_weight_updates",
    "should_quantize_source_weight",
]
