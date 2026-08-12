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
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)

_ENABLED_ENV = "VERL_VLLM_MXFP8_SIM_ENABLED"
_MODE_ENV = "VERL_MXFP8_SIM_MODE"
_VALID_MODES = {"weight_only", "w8a8"}
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


def is_mxfp8_sim_enabled() -> bool:
    return os.environ.get(_ENABLED_ENV, "0") == "1"


def get_mxfp8_sim_mode() -> str:
    mode = os.environ.get(_MODE_ENV, "w8a8").lower()
    if mode not in _VALID_MODES:
        raise ValueError(f"{_MODE_ENV} must be one of {sorted(_VALID_MODES)}, got {mode!r}")
    return mode


def should_quantize_source_weight(name: str, tensor: torch.Tensor) -> bool:
    """Select logical Linear weights before vLLM packing/sharding."""
    if not name.endswith(".weight") or tensor.ndim != 2:
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
    before_tensors = _stats.weight_tensors
    before_elements = _stats.weight_elements
    try:
        for name, tensor in weights:
            if should_quantize_source_weight(name, tensor):
                tensor = _quant_dequant_last_dim(tensor)
                _stats.weight_tensors += 1
                _stats.weight_elements += tensor.numel()
            yield name, tensor
    finally:
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
        if isinstance(module, LinearBase) and "lm_head" not in name.lower():
            setattr(module, _ACTIVATION_MARKER, True)
            marked += 1
    logger.warning("MXFP8 simulation marked %d Linear layers for activation QDQ", marked)
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
    logger.warning("Applied MXFP8 simulation patch (mode=%s)", get_mxfp8_sim_mode())


def get_mxfp8_sim_stats() -> MXFP8SimStats:
    return MXFP8SimStats(**vars(_stats))


__all__ = [
    "apply_mxfp8_sim_patches",
    "get_mxfp8_sim_mode",
    "get_mxfp8_sim_stats",
    "is_mxfp8_sim_enabled",
    "mark_mxfp8_sim_linears",
    "quant_dequant_weight_updates",
    "should_quantize_source_weight",
]
