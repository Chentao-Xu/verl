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

from verl.models.transformers.openpangu_vl import (
    OPENPANGU_MODEL_TYPE,
    apply_openpangu_config_compat,
    apply_openpangu_transformers_compat,
    normalize_openpangu_position_ids,
    patch_openpangu_model_position_ids,
    patch_openpangu_processor,
)
from verl.models.transformers.openpangu_vllm import patch_openpangu_vllm_processing_info
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


def test_patch_openpangu_model_position_ids():
    class FakeOpenPanguModel:
        def forward(self, position_ids=None):
            return position_ids

    model = FakeOpenPanguModel()
    patch_openpangu_model_position_ids(model)
    position_ids = torch.arange(4 * 2 * 8).reshape(4, 2, 8)

    assert torch.equal(model.forward(position_ids=position_ids), position_ids[1:])
