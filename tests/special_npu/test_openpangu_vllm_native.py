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

"""Validate vLLM 0.18 native OpenPangu-VL registration and preprocessing.

Run with:
    python tests/special_npu/test_openpangu_vllm_native.py \
        --model-path /data/models/openPangu-VL-7B
"""

import argparse
import subprocess
import sys
from importlib.metadata import entry_points

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path")
    parser.add_argument("--plugin-worker-check", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def assert_plugin_installed() -> None:
    plugins = {entry_point.name: entry_point.value for entry_point in entry_points(group="vllm.general_plugins")}
    expected = "verl.models.transformers.openpangu_vllm:register"
    if plugins.get("openpangu_vl") != expected:
        raise AssertionError(f"OpenPangu vLLM plugin is not installed: {plugins}")


def assert_weight_mapping(model_class: type) -> None:
    mapper = model_class.hf_to_vllm_mapper
    expected = {
        "model.language_model.layers.0.self_attn.q_proj.weight": (
            "language_model.model.layers.0.self_attn.q_proj.weight"
        ),
        "model.visual.blocks.0.attn.qkv.weight": "visual.blocks.0.attn.qkv.weight",
        "lm_head.weight": "language_model.lm_head.weight",
        "model.embed_tokens.weight": "language_model.model.embed_tokens.weight",
    }
    for source, target in expected.items():
        mapped = mapper.apply_list([source])[0]
        if mapped != target:
            raise AssertionError(f"Weight mapping mismatch: {source} -> {mapped}, expected {target}")


def assert_plugin_loaded_in_subprocess() -> None:
    result = subprocess.run(
        [sys.executable, __file__, "--plugin-worker-check"],
        check=True,
        capture_output=True,
        text=True,
    )
    if "OPENPANGU_VLLM_WORKER_PLUGIN_PASS" not in result.stdout:
        raise AssertionError(f"Plugin subprocess check did not pass: {result.stdout}\n{result.stderr}")


def run_plugin_worker_check() -> None:
    from vllm.plugins import load_general_plugins

    load_general_plugins()

    from vllm.model_executor.models.openpangu_vl import OpenPanguVLProcessingInfo

    if not getattr(OpenPanguVLProcessingInfo, "_verl_openpangu_processor_compat", False):
        raise AssertionError("OpenPangu vLLM plugin was not applied in the subprocess")
    print("OPENPANGU_VLLM_WORKER_PLUGIN_PASS", flush=True)


def main() -> None:
    args = parse_args()
    if args.plugin_worker_check:
        run_plugin_worker_check()
        return
    if args.model_path is None:
        raise ValueError("--model-path is required")

    assert_plugin_installed()
    assert_plugin_loaded_in_subprocess()

    from vllm import ModelRegistry
    from vllm.config import ModelConfig
    from vllm.model_executor.models.interfaces import SupportsMRoPE
    from vllm.model_executor.models.openpangu_vl import OpenPanguVLForConditionalGeneration
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.plugins import load_general_plugins

    load_general_plugins()

    if "pangu_infer" in sys.modules or any(name.startswith("pangu_infer.") for name in sys.modules):
        raise AssertionError("The legacy pangu_infer patch package was imported")

    supported_architectures = set(ModelRegistry.get_supported_archs())
    required_architectures = {"PanguEmbeddedForCausalLM", "OpenPanguVLForConditionalGeneration"}
    missing_architectures = required_architectures - supported_architectures
    if missing_architectures:
        raise AssertionError(f"Missing native vLLM architectures: {sorted(missing_architectures)}")

    model_config = ModelConfig(
        model=args.model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=1024,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 1},
    )
    model_info, architecture = ModelRegistry.inspect_model_cls(model_config.hf_config.architectures, model_config)
    if architecture != "OpenPanguVLForConditionalGeneration":
        raise AssertionError(f"Unexpected architecture: {architecture}")
    if not model_info.supports_multimodal:
        raise AssertionError("OpenPangu vLLM model is not marked as multimodal")
    if SupportsMRoPE not in OpenPanguVLForConditionalGeneration.__mro__:
        raise AssertionError("OpenPangu vLLM model does not implement the native mRoPE interface")

    processor = MULTIMODAL_REGISTRY.create_processor(model_config)
    hf_processor = processor.info.get_hf_processor()
    processor_module = sys.modules[hf_processor.__class__.__module__]
    if not hasattr(processor_module, "np"):
        raise AssertionError("OpenPangu processor NumPy compatibility was not applied")

    placeholder = OpenPanguVLForConditionalGeneration.get_placeholder_str("image", 0)
    image = Image.new("RGB", (224, 224), color=(64, 128, 192))
    mm_items = processor.info.parse_mm_data({"image": image})
    processed = processor(placeholder, mm_items=mm_items, hf_processor_mm_kwargs={})

    dummy_inputs = processor.dummy_inputs.get_dummy_processor_inputs(
        seq_len=model_config.max_model_len,
        mm_counts={"image": 1},
        mm_options={},
    )
    expected_placeholder = f"{hf_processor.vision_start_token}{hf_processor.image_token}{hf_processor.vision_end_token}"
    if placeholder != expected_placeholder:
        raise AssertionError(f"Unexpected engine placeholder: {placeholder}")
    if dummy_inputs.prompt != hf_processor.image_token or len(dummy_inputs.mm_data_items["image"]) != 1:
        raise AssertionError("OpenPangu dummy image inputs do not match the native placeholder")

    image_grid_thw = processed["mm_kwargs"]["image"][0]["image_grid_thw"].data
    expected_image_tokens = int(image_grid_thw.prod()) // 4
    image_token_id = hf_processor.tokenizer.get_vocab()[hf_processor.image_token]
    actual_image_tokens = processed["prompt_token_ids"].count(image_token_id)
    if actual_image_tokens != expected_image_tokens:
        raise AssertionError(
            f"Image token count mismatch: got {actual_image_tokens}, expected {expected_image_tokens}"
        )

    assert_weight_mapping(OpenPanguVLForConditionalGeneration)
    print(
        "OPENPANGU_VLLM_NATIVE_PASS "
        f"architecture={architecture} processor={type(processor).__name__} "
        f"image_grid_thw={image_grid_thw.tolist()} image_tokens={actual_image_tokens}",
        flush=True,
    )


if __name__ == "__main__":
    main()
