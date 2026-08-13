import pytest
import torch

from verl.utils.vllm.mxfp8_sim import (
    classify_mxfp8_activation_module,
    get_mxfp8_activation_groups,
    should_quantize_activation_module,
    should_quantize_source_weight,
)


def test_activation_module_classification():
    assert classify_mxfp8_activation_module("model.layers.0.self_attn.qkv_proj") == "qkv"
    assert classify_mxfp8_activation_module("model.layers.0.self_attn.o_proj") == "o"
    assert classify_mxfp8_activation_module("model.layers.0.mlp.gate_up_proj") == "gate_up"
    assert classify_mxfp8_activation_module("model.layers.0.mlp.down_proj") == "down"
    assert classify_mxfp8_activation_module("model.layers.0.foo") == "other"


def test_activation_group_filter(monkeypatch):
    monkeypatch.setenv("VERL_MXFP8_SIM_SCOPE", "language_only")
    monkeypatch.setenv("VERL_MXFP8_SIM_ACTIVATION_GROUPS", "qkv,down")
    assert get_mxfp8_activation_groups() == {"qkv", "down"}
    assert should_quantize_activation_module("model.layers.0.self_attn.qkv_proj")
    assert should_quantize_activation_module("model.layers.0.mlp.down_proj")
    assert not should_quantize_activation_module("model.layers.0.mlp.gate_up_proj")
    assert not should_quantize_activation_module("visual.blocks.0.attn.qkv_proj")


def test_mxfp8_qat_modes_are_registered():
    pytest.importorskip("triton")
    from verl.utils.qat.linear import QATMode

    assert QATMode("mxfp8_w8a16") is QATMode.MXFP8_W8A16
    assert QATMode("mxfp8_w8a8") is QATMode.MXFP8_W8A8


def test_should_quantize_openpangu_linear_weights(monkeypatch):
    monkeypatch.setenv("VERL_MXFP8_SIM_SCOPE", "all")
    matrix = torch.empty(64, 32)

    assert should_quantize_source_weight("model.layers.0.self_attn.q_proj.weight", matrix)
    assert should_quantize_source_weight("model.layers.0.mlp.down_proj.weight", matrix)
    assert should_quantize_source_weight("visual.blocks.0.attn.qkv.weight", matrix)
    assert should_quantize_source_weight("visual.merger.0.mlp.0.weight", matrix)


def test_should_keep_non_linear_weights_in_bf16(monkeypatch):
    monkeypatch.setenv("VERL_MXFP8_SIM_SCOPE", "all")
    matrix = torch.empty(64, 32)

    assert not should_quantize_source_weight("model.embed_tokens.weight", matrix)
    assert not should_quantize_source_weight("lm_head.weight", matrix)
    assert not should_quantize_source_weight("visual.blocks.0.norm1.weight", torch.empty(32))
    assert not should_quantize_source_weight("visual.patch_embed.proj.weight", torch.empty(32, 3, 2, 2, 2))


@pytest.mark.parametrize(
    ("scope", "language_expected", "visual_expected"),
    [
        ("all", True, True),
        ("language_only", True, False),
        ("visual_only", False, True),
    ],
)
def test_should_honor_model_scope(monkeypatch, scope, language_expected, visual_expected):
    monkeypatch.setenv("VERL_MXFP8_SIM_SCOPE", scope)
    matrix = torch.empty(64, 32)

    assert should_quantize_source_weight("model.layers.0.self_attn.q_proj.weight", matrix) is language_expected
    assert should_quantize_source_weight("model.visual.blocks.0.attn.qkv.weight", matrix) is visual_expected


def test_should_honor_language_layer_range(monkeypatch):
    monkeypatch.setenv("VERL_MXFP8_SIM_SCOPE", "language_only")
    monkeypatch.setenv("VERL_MXFP8_SIM_LANGUAGE_LAYER_RANGE", "3:7")
    matrix = torch.empty(64, 32)

    assert not should_quantize_source_weight("model.layers.2.self_attn.q_proj.weight", matrix)
    assert should_quantize_source_weight("model.layers.3.self_attn.q_proj.weight", matrix)
    assert should_quantize_source_weight("model.layers.7.mlp.down_proj.weight", matrix)
    assert not should_quantize_source_weight("model.layers.8.mlp.down_proj.weight", matrix)
    assert not should_quantize_source_weight("visual.blocks.3.attn.qkv.weight", matrix)
