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

"""Narrow Ascend compatibility bridges for vLLM's native OpenPangu-VL."""

from functools import wraps
from types import ModuleType

from verl.models.transformers.openpangu_vl import patch_openpangu_processor


def patch_openpangu_vllm_processing_info(processing_info_class: type) -> None:
    """Patch processors returned by vLLM without changing vLLM model code."""
    if getattr(processing_info_class, "_verl_openpangu_processor_compat", False):
        return

    original_get_hf_processor = processing_info_class.get_hf_processor

    @wraps(original_get_hf_processor)
    def get_hf_processor(self, *args, **kwargs):
        processor = original_get_hf_processor(self, *args, **kwargs)
        patch_openpangu_processor(processor)
        return processor

    processing_info_class.get_hf_processor = get_hf_processor
    processing_info_class._verl_openpangu_processor_compat = True


def patch_openpangu_vllm_vision_backend(model_module: ModuleType, flash_backend) -> None:
    """Allow the Ascend OOT vision attention implementation past a stale guard."""
    vision_class = model_module.OpenPanguVisionTransformer
    if getattr(vision_class, "_verl_ascend_backend_compat", False):
        return

    original_init = vision_class.__init__

    @wraps(original_init)
    def init(self, *args, **kwargs):
        original_get_vit_attn_backend = model_module.get_vit_attn_backend
        model_module.get_vit_attn_backend = lambda *args, **kwargs: flash_backend
        try:
            original_init(self, *args, **kwargs)
        finally:
            model_module.get_vit_attn_backend = original_get_vit_attn_backend

    vision_class.__init__ = init
    vision_class._verl_ascend_backend_compat = True


def register() -> None:
    """vLLM general plugin entry point, loaded in frontend and worker processes."""
    try:
        from vllm.model_executor.models import openpangu_vl
        from vllm.platforms import current_platform
        from vllm.v1.attention.backends.registry import AttentionBackendEnum
    except ImportError:
        return

    patch_openpangu_vllm_processing_info(openpangu_vl.OpenPanguVLProcessingInfo)
    if current_platform.device_type == "npu":
        patch_openpangu_vllm_vision_backend(openpangu_vl, AttentionBackendEnum.FLASH_ATTN)


__all__ = ["patch_openpangu_vllm_processing_info", "patch_openpangu_vllm_vision_backend", "register"]
