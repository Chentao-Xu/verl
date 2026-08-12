import torch

from verl.utils.vllm.mxfp8_sim import should_quantize_source_weight


def test_should_quantize_openpangu_linear_weights():
    matrix = torch.empty(64, 32)

    assert should_quantize_source_weight("model.layers.0.self_attn.q_proj.weight", matrix)
    assert should_quantize_source_weight("model.layers.0.mlp.down_proj.weight", matrix)
    assert should_quantize_source_weight("visual.blocks.0.attn.qkv.weight", matrix)
    assert should_quantize_source_weight("visual.merger.0.mlp.0.weight", matrix)


def test_should_keep_non_linear_weights_in_bf16():
    matrix = torch.empty(64, 32)

    assert not should_quantize_source_weight("model.embed_tokens.weight", matrix)
    assert not should_quantize_source_weight("lm_head.weight", matrix)
    assert not should_quantize_source_weight("visual.blocks.0.norm1.weight", torch.empty(32))
    assert not should_quantize_source_weight("visual.patch_embed.proj.weight", torch.empty(32, 3, 2, 2, 2))
