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

"""Single-NPU OpenPangu actor load, forward, and backward smoke test.

Run with:
    python tests/special_npu/test_openpangu_actor_smoke.py \
        --model-path /data/models/openPangu-VL-7B
"""

import argparse
import gc

import torch
import torch_npu  # noqa: F401
from PIL import Image

from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.utils.model import get_hf_auto_model_class
from verl.workers.config.model import HFModelConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="npu:0")
    return parser.parse_args()


def make_conversation(include_image: bool) -> list[dict]:
    content = []
    if include_image:
        content.append({"type": "image"})
    content.append({"type": "text", "text": "请用一句话介绍你自己。"})
    return [{"role": "user", "content": content}]


def prepare_inputs(processor, include_image: bool) -> dict[str, torch.Tensor]:
    conversation = make_conversation(include_image)
    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    images = [Image.new("RGB", (224, 224), color=(64, 128, 192))] if include_image else None
    inputs = processor(text=[text], images=images, padding=False, return_tensors="pt")

    rope_result = processor.get_rope_index(
        input_ids=inputs["input_ids"],
        image_grid_thw=inputs.get("image_grid_thw"),
        video_grid_thw=inputs.get("video_grid_thw"),
        attention_mask=inputs["attention_mask"],
    )
    vision_position_ids = rope_result[0] if isinstance(rope_result, tuple) else rope_result
    text_position_ids = torch.arange(inputs["input_ids"].shape[-1], dtype=torch.long).view(1, 1, -1)
    text_position_ids = text_position_ids.expand(1, inputs["input_ids"].shape[0], -1)
    inputs["position_ids"] = torch.cat((text_position_ids, vision_position_ids), dim=0)
    return dict(inputs)


def move_to_device(inputs: dict, device: str) -> dict:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}


def assert_finite_gradients(model) -> tuple[int, int]:
    gradient_count = 0
    visual_gradient_count = 0
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        gradient_count += 1
        if name.startswith("model.visual"):
            visual_gradient_count += 1
        if not torch.isfinite(parameter.grad).all():
            raise AssertionError(f"Non-finite gradient in {name}")
    if gradient_count == 0:
        raise AssertionError("No model gradients were produced")
    if visual_gradient_count == 0:
        raise AssertionError("No visual encoder gradients were produced")
    return gradient_count, visual_gradient_count


def main() -> None:
    args = parse_args()
    torch.npu.set_device(args.device)

    model_config = HFModelConfig(
        path=args.model_path,
        trust_remote_code=True,
        override_config={"attn_implementation": "eager"},
    )
    auto_class = get_hf_auto_model_class(model_config.hf_config)
    model, loading_info = auto_class.from_pretrained(
        model_config.local_path,
        config=model_config.hf_config,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        output_loading_info=True,
    )

    if loading_info["missing_keys"] or loading_info["unexpected_keys"] or loading_info["mismatched_keys"]:
        raise AssertionError(f"Weight loading issues: {loading_info}")
    if not hasattr(model, "lm_head"):
        raise AssertionError("OpenPangu actor is missing lm_head")

    apply_monkey_patch(model=model, use_remove_padding=False)
    model = model.to(args.device)
    model.train()

    text_inputs = move_to_device(prepare_inputs(model_config.processor, include_image=False), args.device)
    with torch.no_grad():
        text_outputs = model(**text_inputs, use_cache=False)
    if text_outputs.logits.shape[:2] != text_inputs["input_ids"].shape:
        raise AssertionError(
            f"Text logits shape {tuple(text_outputs.logits.shape)} does not match "
            f"input shape {tuple(text_inputs['input_ids'].shape)}"
        )
    print(f"OPENPANGU_TEXT_FORWARD_PASS logits_shape={tuple(text_outputs.logits.shape)}", flush=True)

    image_inputs = move_to_device(prepare_inputs(model_config.processor, include_image=True), args.device)
    image_outputs = model(
        **image_inputs,
        labels=image_inputs["input_ids"],
        use_cache=False,
    )
    if image_outputs.loss is None or not torch.isfinite(image_outputs.loss):
        raise AssertionError(f"Invalid image-text loss: {image_outputs.loss}")
    print(
        f"OPENPANGU_IMAGE_FORWARD_PASS logits_shape={tuple(image_outputs.logits.shape)} "
        f"loss={image_outputs.loss.item():.6f}",
        flush=True,
    )

    image_outputs.loss.backward()
    gradient_count, visual_gradient_count = assert_finite_gradients(model)
    torch.npu.synchronize()
    print(
        f"OPENPANGU_BACKWARD_PASS gradients={gradient_count} visual_gradients={visual_gradient_count}",
        flush=True,
    )

    del image_outputs, image_inputs, text_outputs, text_inputs, model
    gc.collect()
    torch.npu.empty_cache()
    print("OPENPANGU_ACTOR_SMOKE_PASS", flush=True)


if __name__ == "__main__":
    main()
