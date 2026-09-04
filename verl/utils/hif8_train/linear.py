"""NPU delayed-HiF8 QDQ simulation for the three GEMMs of a training Linear.

Parameters and optimizer states remain BF16. Each matrix multiplication
quantizes and dequantizes both operands with Huawei's ``dhif8`` simulator,
then executes the matmul in BF16. This is a numerical simulator, not QAT.
"""

import logging
import re

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _quant_dequant(x: torch.Tensor, dim: int) -> torch.Tensor:
    if x.device.type != "npu":
        raise RuntimeError(f"HiF8 Actor training simulation requires NPU tensors, got {x.device}")

    from quant_cy_npu.QTensor import delay_quant_dequant_float
    from quant_cy_npu.QType import QType

    contiguous = x.contiguous()
    amax = float(contiguous.detach().abs().max())
    return delay_quant_dequant_float(
        contiguous,
        amax,
        15,
        QType("dhif8").dim_(dim),
        force_py=False,
    )


def _hif8_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Simulate delayed HiF8 A@B on both reduction dimensions."""
    return torch.matmul(_quant_dequant(a, -1), _quant_dequant(b, 0))


class _HIF8LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        input_shape = x.shape
        x_2d = x.reshape(-1, input_shape[-1])
        output = _hif8_matmul(x_2d, weight.t().contiguous())
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

        grad_input = _hif8_matmul(grad_2d, weight)
        grad_weight = _hif8_matmul(grad_2d.t().contiguous(), x_2d)
        grad_bias = grad_2d.sum(dim=0) if ctx.has_bias else None
        return grad_input.reshape(ctx.input_shape), grad_weight, grad_bias


class HIF8TrainLinear(nn.Linear):
    """Drop-in ``nn.Linear`` whose forward/backward GEMMs use delayed HiF8 QDQ."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _HIF8LinearFunction.apply(x, self.weight, self.bias)


_LANGUAGE_LAYER_RE = re.compile(r"(?:^|\.)(?:language_model\.)?(?:model\.)?layers\.(\d+)\.")


def _language_layer_index(name: str) -> int | None:
    match = _LANGUAGE_LAYER_RE.search(name)
    return int(match.group(1)) if match else None


def apply_hif8_train_sim(model: nn.Module, layer_start: int = 0, layer_end: int = 33) -> int:
    """Convert OpenPangu language Linear layers in ``[layer_start, layer_end]``."""
    if layer_start < 0 or layer_end < layer_start:
        raise ValueError(f"Invalid HiF8 Actor language layer range: {layer_start}:{layer_end}")

    converted = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or isinstance(module, HIF8TrainLinear):
            continue
        layer_idx = _language_layer_index(name)
        if layer_idx is None or not layer_start <= layer_idx <= layer_end:
            continue
        if any(excluded in name for excluded in ("visual", "lm_head", "embed_tokens")):
            continue
        module.__class__ = HIF8TrainLinear
        converted.append(name)

    if not converted:
        raise RuntimeError(
            "HiF8 Actor training simulation did not find language Linear layers; "
            "check model module names and requested layer range"
        )
    logger.info(
        "Enabled non-QAT HiF8 Actor simulation on %d language Linear layers (range %d:%d)",
        len(converted),
        layer_start,
        layer_end,
    )
    return len(converted)
