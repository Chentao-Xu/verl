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

import sys
from types import ModuleType, SimpleNamespace

import torch
import transformers.utils as transformers_utils
from transformers import AutoModelForCausalLM
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from verl.experimental.agent_loop.agent_loop import AgentLoopWorker
from verl.models.transformers.openpangu_vl import (
    OPENPANGU_MODEL_TYPE,
    apply_openpangu_config_compat,
    apply_openpangu_transformers_compat,
    disable_openpangu_npu_fused_infer_attention,
    normalize_openpangu_processor_text,
    normalize_openpangu_position_ids,
    patch_openpangu_model_position_ids,
    patch_openpangu_npu_attention_mask,
    patch_openpangu_processor,
    reinitialize_openpangu_rotary_embeddings,
)
from verl.models.transformers.openpangu_vllm import (
    patch_openpangu_vllm_processing_info,
    patch_openpangu_vllm_vision_backend,
)
from verl.utils.model import get_hf_auto_model_class


def test_get_hf_auto_model_class_uses_openpangu_causal_lm():
    config = SimpleNamespace(
        model_type=OPENPANGU_MODEL_TYPE,
        architectures=["OpenPanguVLForConditionalGeneration"],
        auto_map={
            "AutoModel": "modeling_openpangu_vl.OpenPanguVLModel",
            "AutoModelForCausalLM": "modeling_openpangu_vl.OpenPanguVL",
        },
    )

    assert get_hf_auto_model_class(config) is AutoModelForCausalLM


def test_apply_openpangu_transformers_compat_exports_loss_kwargs(monkeypatch):
    monkeypatch.delattr(transformers_utils, "LossKwargs", raising=False)
    monkeypatch.delitem(ROPE_INIT_FUNCTIONS, "default", raising=False)

    apply_openpangu_transformers_compat()

    assert transformers_utils.LossKwargs is transformers_utils.TransformersKwargs
    assert callable(ROPE_INIT_FUNCTIONS["default"])


def test_apply_openpangu_config_compat():
    config = SimpleNamespace(
        text_config=SimpleNamespace(initializer_range=0.02),
        vision_config=SimpleNamespace(),
        pad_token_id=45892,
        bos_token_id=1,
        eos_token_id=45892,
    )

    apply_openpangu_config_compat(config)

    assert config.vision_config.initializer_range == 0.02
    assert config.text_config.pad_token_id == 45892
    assert config.text_config.bos_token_id == 1
    assert config.text_config.eos_token_id == 45892


def test_normalize_openpangu_position_ids():
    position_ids = torch.arange(4 * 2 * 8).reshape(4, 2, 8)

    normalized = normalize_openpangu_position_ids(position_ids)

    assert normalized.shape == (3, 2, 8)
    assert torch.equal(normalized, position_ids[1:])


def test_normalize_openpangu_processor_text_collapses_expanded_image_tokens():
    processor = SimpleNamespace(
        image_token="[unused19]",
        vision_start_token="[unused18]",
        vision_end_token="[unused20]",
    )
    text = [
        "before[unused18]" + "[unused19]" * 4 + "[unused20]after",
        "[unused18][unused19][unused20]second",
    ]

    normalized = normalize_openpangu_processor_text(processor, text)

    assert normalized == [
        "before[unused18][unused19][unused20]after",
        "[unused18][unused19][unused20]second",
    ]
    assert normalized is not text


def test_agent_loop_omits_unsupported_mm_token_type_ids_for_openpangu():
    class FakeProcessor:
        image_token_id = 19
        video_token_id = 32

        def get_rope_index(self, input_ids, attention_mask, image_grid_thw=None, video_grid_thw=None):
            del image_grid_thw, video_grid_thw
            return torch.zeros((3, input_ids.shape[0], input_ids.shape[1]), dtype=torch.long), None

    worker = SimpleNamespace(processor=FakeProcessor())
    input_ids = torch.tensor([[1, 19, 19, 2]])
    attention_mask = torch.ones_like(input_ids)
    multi_modal_inputs = {
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
        "mm_token_type_ids": torch.tensor([[0, 1, 1, 0]]),
    }

    position_ids = AgentLoopWorker._compute_position_ids(
        worker,
        input_ids,
        attention_mask,
        multi_modal_inputs,
    )

    assert position_ids.shape == (1, 4, 4)
    assert "mm_token_type_ids" not in multi_modal_inputs


def test_patch_openpangu_processor(monkeypatch):
    module_name = "fake_openpangu_processor"
    processor_module = ModuleType(module_name)
    monkeypatch.setitem(sys.modules, module_name, processor_module)
    processor_class = type("FakeOpenPanguProcessor", (), {"__module__": module_name})

    patch_openpangu_processor(processor_class())

    assert processor_module.np.__name__ == "numpy"


def test_patch_openpangu_vllm_processing_info(monkeypatch):
    module_name = "fake_openpangu_vllm_processor"
    processor_module = ModuleType(module_name)
    monkeypatch.setitem(sys.modules, module_name, processor_module)
    processor_class = type("FakeOpenPanguProcessor", (), {"__module__": module_name})

    class FakeProcessingInfo:
        def get_hf_processor(self, marker=None):
            assert marker == "expected"
            return processor_class()

    original_get_hf_processor = FakeProcessingInfo.get_hf_processor
    patch_openpangu_vllm_processing_info(FakeProcessingInfo)
    patched_get_hf_processor = FakeProcessingInfo.get_hf_processor
    processor = FakeProcessingInfo().get_hf_processor(marker="expected")

    assert patched_get_hf_processor is not original_get_hf_processor
    assert processor_module.np.__name__ == "numpy"

    patch_openpangu_vllm_processing_info(FakeProcessingInfo)
    assert FakeProcessingInfo.get_hf_processor is patched_get_hf_processor


def test_patch_openpangu_vllm_vision_backend():
    observed_backends = []
    original_backend = object()
    flash_backend = object()

    class FakeVisionTransformer:
        def __init__(self):
            observed_backends.append(model_module.get_vit_attn_backend())

    model_module = SimpleNamespace(
        OpenPanguVisionTransformer=FakeVisionTransformer,
        get_vit_attn_backend=lambda: original_backend,
    )

    patch_openpangu_vllm_vision_backend(model_module, flash_backend)
    patched_init = FakeVisionTransformer.__init__
    FakeVisionTransformer()

    assert observed_backends == [flash_backend]
    assert model_module.get_vit_attn_backend() is original_backend

    patch_openpangu_vllm_vision_backend(model_module, flash_backend)
    assert FakeVisionTransformer.__init__ is patched_init


def test_patch_openpangu_model_position_ids():
    class FakeOpenPanguModel:
        def forward(self, position_ids=None):
            return position_ids

    model = FakeOpenPanguModel()
    patch_openpangu_model_position_ids(model)
    position_ids = torch.arange(4 * 2 * 8).reshape(4, 2, 8)

    assert torch.equal(model.forward(position_ids=position_ids), position_ids[1:])


def test_reinitialize_openpangu_rotary_embeddings():
    class OpenPanguVLRotaryEmbedding(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = object()
            self.register_buffer("inv_freq", torch.zeros(2), persistent=False)
            self.original_inv_freq = self.inv_freq
            self.attention_scaling = 0.0

        @staticmethod
        def rope_init_fn(config, device):
            del config
            return torch.tensor([1.0, 0.5], device=device), 1.0

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rotary = OpenPanguVLRotaryEmbedding()
            self.vision_rotary = OpenPanguVisionRotaryEmbedding()

    class OpenPanguVisionRotaryEmbedding(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("inv_freq", torch.zeros(2), persistent=False)

    model = FakeModel()
    reinitialize_openpangu_rotary_embeddings(model)

    assert torch.equal(model.rotary.inv_freq, torch.tensor([1.0, 0.5]))
    assert model.rotary.original_inv_freq is model.rotary.inv_freq
    assert model.rotary.attention_scaling == 1.0
    assert torch.allclose(model.vision_rotary.inv_freq, torch.tensor([1.0, 0.01]))


def test_disable_openpangu_npu_fused_infer_attention(monkeypatch):
    class PanguEmbeddedAttention(torch.nn.Module):
        def forward(self, attention_mask=None):
            if not self.training and NPU_ATTN_INFR:
                return ~attention_mask.bool()
            return attention_mask

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = PanguEmbeddedAttention()

    class OpenPanguVLVisionAttention(torch.nn.Module):
        pass

    class FakeMultimodalModel(FakeModel):
        def __init__(self):
            super().__init__()
            self.vision_attention = OpenPanguVLVisionAttention()

    monkeypatch.setitem(
        sys.modules[PanguEmbeddedAttention.__module__].__dict__,
        "NPU_ATTN_INFR",
        True,
    )
    monkeypatch.setitem(
        sys.modules[OpenPanguVLVisionAttention.__module__].__dict__,
        "NPU_ATTN_INFR",
        True,
    )
    model = FakeMultimodalModel().eval()
    additive_mask = torch.tensor([[0.0, torch.finfo(torch.float32).min], [0.0, 0.0]])

    disable_openpangu_npu_fused_infer_attention(model)

    assert torch.equal(model.attention(attention_mask=additive_mask), additive_mask)
    assert NPU_ATTN_INFR is False
    model.train()
    assert torch.equal(model.attention(attention_mask=additive_mask), additive_mask)
