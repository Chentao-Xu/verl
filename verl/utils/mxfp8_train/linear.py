"""NPU MXFP8 QDQ simulation for the three GEMMs of a training Linear.

Parameters and optimizer states remain in their normal training dtype.  Each
matrix multiplication quantizes and dequantizes both operands to MXFP8 E4M3
along its reduction dimension, then executes the matmul in the original dtype.
This mirrors torchao MXLinear's forward/backward numerical semantics without
requiring a native low-precision GEMM or a straight-through fake quantizer.
"""

import logging
import re

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _quant_dequant(x: torch.Tensor, dim: int) -> torch.Tensor:
    if x.device.type != "npu":
        raise RuntimeError(f"MXFP8 Actor training simulation requires NPU tensors, got {x.device}")

    from quant_cy_npu.QTensor import quant_dequant_float
    from quant_cy_npu.QType import QType

    return quant_dequant_float(x.contiguous(), QType("mxfp8e4m3").dim_(dim), force_py=False)


def _mxfp8_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Simulate MXFP8 A@B; the reduction dimension is A[-1] and B[0]."""
    return torch.matmul(_quant_dequant(a, -1), _quant_dequant(b, 0))


class _MXFP8LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        input_shape = x.shape
        x_2d = x.reshape(-1, input_shape[-1])
        output = _mxfp8_matmul(x_2d, weight.t().contiguous())
        if bias is not None:
            output = output + bias
        ctx.save_for_backward(x_2d, weight)
        ctx.input_shape = input_shape
        ctx.has_bias = bias is not None
        return output.reshape(*input_shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_2d, weight = ctx.saved_tensors
        grad_2d = grad_output.reshape(-1, grad_output.shape[-1])

        grad_input = _mxfp8_matmul(grad_2d, weight)
        grad_weight = _mxfp8_matmul(grad_2d.t().contiguous(), x_2d)
        grad_bias = grad_2d.sum(dim=0) if ctx.has_bias else None
        return grad_input.reshape(ctx.input_shape), grad_weight, grad_bias


class MXFP8TrainLinear(nn.Linear):
    """Drop-in ``nn.Linear`` whose forward and backward GEMMs use MXFP8 QDQ."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _MXFP8LinearFunction.apply(x, self.weight, self.bias)


_LANGUAGE_LAYER_RE = re.compile(r"(?:^|\.)(?:language_model\.)?(?:model\.)?layers\.(\d+)\.")


def _language_layer_index(name: str) -> int | None:
    match = _LANGUAGE_LAYER_RE.search(name)
    return int(match.group(1)) if match else None


def apply_mxfp8_train_sim(model: nn.Module, layer_start: int = 0, layer_end: int = 33) -> int:
    """Convert OpenPangu language Linear layers in ``[layer_start, layer_end]``.

    Conversion changes the Python class in place, preserving Parameters,
    state-dict keys, dtype, device, and checkpoint compatibility.
    """
    if layer_start < 0 or layer_end < layer_start:
        raise ValueError(f"Invalid MXFP8 Actor language layer range: {layer_start}:{layer_end}")

    converted = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or isinstance(module, MXFP8TrainLinear):
            continue
        layer_idx = _language_layer_index(name)
        if layer_idx is None or not layer_start <= layer_idx <= layer_end:
            continue
        if any(excluded in name for excluded in ("visual", "lm_head", "embed_tokens")):
            continue
        module.__class__ = MXFP8TrainLinear
        converted.append(name)

    if not converted:
        raise RuntimeError(
            "MXFP8 Actor training simulation did not find any language Linear layers; "
            "check the model module names and requested layer range"
        )
    logger.info(
        "Enabled non-QAT MXFP8 Actor simulation on %d language Linear layers (range %d:%d)",
        len(converted),
        layer_start,
        layer_end,
    )
    return len(converted)
