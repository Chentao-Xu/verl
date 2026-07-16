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

"""Narrow compatibility bridge for vLLM's native OpenPangu-VL processor."""

from functools import wraps

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


def register() -> None:
    """vLLM general plugin entry point, loaded in frontend and worker processes."""
    try:
        from vllm.model_executor.models.openpangu_vl import OpenPanguVLProcessingInfo
    except ImportError:
        return

    patch_openpangu_vllm_processing_info(OpenPanguVLProcessingInfo)


__all__ = ["patch_openpangu_vllm_processing_info", "register"]
