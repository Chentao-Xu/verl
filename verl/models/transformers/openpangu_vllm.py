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

"""Register the OpenPangu-VL runtime adapter supplied with the checkpoint."""

import os
from pathlib import Path


def _prepend_pythonpath(path: Path) -> None:
    entries = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    value = str(path)
    if value not in entries:
        os.environ["PYTHONPATH"] = os.pathsep.join([value, *filter(None, entries)])


def _apply_vllm_011_api_compat() -> None:
    """Expose APIs that moved after vLLM 0.11 for the checkpoint adapter."""
    import vllm.sequence as sequence
    from vllm.entrypoints import chat_utils
    from vllm.v1.attention.backends import utils as v1_attention_utils

    if not hasattr(v1_attention_utils, "validate_kv_sharing_target"):
        from vllm.attention.utils.kv_sharing_utils import validate_kv_sharing_target

        v1_attention_utils.validate_kv_sharing_target = validate_kv_sharing_target

    if not hasattr(chat_utils, "random_tool_call_id"):
        chat_utils.random_tool_call_id = chat_utils.make_tool_call_id

    if not hasattr(sequence, "Logprob"):
        from vllm.logprobs import Logprob

        sequence.Logprob = Logprob


def register_openpangu_vllm() -> None:
    try:
        import pangu_infer
    except ImportError as exc:
        raise ImportError(
            "OpenPangu vLLM adapter is not importable. Add "
            "<model>/inference/vllm_ascend to PYTHONPATH before starting Verl."
        ) from exc

    # Keep the adapter importable in forked vLLM workers. Do not add the
    # package directory itself: its sitecustomize installs server-wide Pangu
    # patches for MoE and reasoning features that Verl rollouts do not use.
    package_dir = Path(next(iter(pangu_infer.__path__))).resolve()
    _prepend_pythonpath(package_dir.parent)

    _apply_vllm_011_api_compat()

    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "PanguEmbeddedForCausalLM",
        "pangu_infer.models.vllm_ascend.modeling_openpangu_embedded:PanguEmbeddedForCausalLM",
    )
    ModelRegistry.register_model(
        "OpenPanguVLForConditionalGeneration",
        "pangu_infer.models.vllm_ascend.modeling_openpangu_vl:OpenPanguVLForConditionalGeneration",
    )


register_openpangu_vllm()


__all__ = ["register_openpangu_vllm"]
