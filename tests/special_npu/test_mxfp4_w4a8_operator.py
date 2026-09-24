import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from quant_cy_npu.QTensor import quant_dequant_float
from quant_cy_npu.QType import QType


@pytest.mark.parametrize("shape", [(4, 32), (4, 37)])
@pytest.mark.parametrize("qtype_name", ["mxfp4", "mxfp8e4m3", "hifx4"])
def test_mxfp4_w4a8_qdq_is_finite_deterministic_and_shape_preserving(shape, qtype_name):
    torch.manual_seed(42)
    source = torch.randn(shape, dtype=torch.bfloat16, device="npu")
    qtype = QType(qtype_name).dim_(-1)

    first = quant_dequant_float(source.clone(), qtype, force_py=False)
    second = quant_dequant_float(source.clone(), qtype, force_py=False)
    torch.npu.synchronize()

    assert first.shape == source.shape
    assert first.dtype == source.dtype
    assert torch.isfinite(first).all().item()
    torch.testing.assert_close(first, second, rtol=0, atol=0)

    source_fp32 = source.float()
    error = (first.float() - source_fp32).pow(2).mean()
    reference_power = source_fp32.pow(2).mean()
    nmse = (error / reference_power).item()
    assert 0 <= nmse < 1
