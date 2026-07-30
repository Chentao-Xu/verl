# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Compare OpenPangu HF and native-vLLM multimodal preprocessing exactly.

This test does not load model weights. It verifies the contract that must hold
before comparing actor and rollout log probabilities: rendered prompts, token
IDs, image grids, and image-token counts must be identical.
"""

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf
from vllm import ModelRegistry
from vllm.config import ModelConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.plugins import load_general_plugins

from verl.utils import hf_processor
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.tokenizer import build_multimodal_processor_inputs, normalize_token_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--multi-image-samples", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-images", type=int, default=8)
    parser.add_argument("--result-path")
    return parser.parse_args()


def make_dataset(dataset_path: str, processor) -> RLHFDataset:
    config = OmegaConf.create(
        {
            "prompt_key": "prompt",
            "image_key": "images",
            "video_key": "videos",
            "audio_key": "audios",
            "max_prompt_length": 4096,
            "filter_overlong_prompts": False,
            "truncation": "error",
            "shuffle": False,
        }
    )
    return RLHFDataset(dataset_path, processor.tokenizer, config, processor=processor)


def select_indices(dataset: RLHFDataset, samples: int, multi_image_samples: int) -> list[int]:
    if samples <= 0:
        raise ValueError("samples must be positive")
    multi_target = min(samples, max(0, multi_image_samples))
    single_target = samples - multi_target
    single_indices = []
    multi_indices = []
    for index, row in enumerate(dataset.dataframe):
        target = multi_indices if len(row.get("images") or []) > 1 else single_indices
        limit = multi_target if target is multi_indices else single_target
        if len(target) < limit:
            target.append(index)
        if len(single_indices) == single_target and len(multi_indices) == multi_target:
            break
    indices = single_indices + multi_indices
    if len(indices) != samples:
        raise AssertionError(f"Could select only {len(indices)} of {samples} requested samples")
    return indices


def flatten_vllm_image_grids(processed: dict) -> list[list[int]]:
    grids = []
    for item in processed["mm_kwargs"].get("image", []):
        value = item.get("image_grid_thw")
        if value is None:
            continue
        data = value.data
        grids.extend(data.tolist() if data.ndim == 2 else [data.tolist()])
    return grids


def main() -> None:
    args = parse_args()
    load_general_plugins()
    processor = hf_processor(args.model_path, trust_remote_code=True, use_fast=True)
    dataset = make_dataset(args.dataset_path, processor)

    model_config = ModelConfig(
        model=args.model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        enforce_eager=True,
        limit_mm_per_prompt={"image": args.max_images, "video": 0},
    )
    _, architecture = ModelRegistry.inspect_model_cls(model_config.hf_config.architectures, model_config)
    if architecture != "OpenPanguVLForConditionalGeneration":
        raise AssertionError(f"Unexpected vLLM architecture: {architecture}")
    vllm_processor = MULTIMODAL_REGISTRY.create_processor(model_config)

    results = []
    indices = select_indices(dataset, args.samples, args.multi_image_samples)
    for index in indices:
        sample = dataset[index]
        messages = sample["raw_prompt"]
        rendered = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        images, videos, audios = dataset._process_multi_modal_info(
            messages, image_patch_size=dataset.image_patch_size, config=dataset.config
        )
        if videos or audios:
            raise AssertionError("This parity test currently accepts image-only samples")

        hf_inputs = build_multimodal_processor_inputs(processor, text=[rendered], images=images)
        hf_ids = normalize_token_ids(hf_inputs["input_ids"])
        hf_grids = hf_inputs["image_grid_thw"].tolist()

        mm_items = vllm_processor.info.parse_mm_data({"image": images})
        vllm_inputs = vllm_processor(rendered, mm_items=mm_items, hf_processor_mm_kwargs={})
        vllm_ids = vllm_inputs["prompt_token_ids"]
        vllm_grids = flatten_vllm_image_grids(vllm_inputs)

        if hf_ids != vllm_ids:
            mismatch = next((i for i, pair in enumerate(zip(hf_ids, vllm_ids)) if pair[0] != pair[1]), None)
            raise AssertionError(
                f"sample={index} token mismatch at {mismatch}; "
                f"hf_len={len(hf_ids)} vllm_len={len(vllm_ids)}"
            )
        if hf_grids != vllm_grids:
            raise AssertionError(f"sample={index} image grids differ: hf={hf_grids}, vllm={vllm_grids}")

        image_token_id = processor.tokenizer.get_vocab()[processor.image_token]
        image_tokens = hf_ids.count(image_token_id)
        expected_image_tokens = sum(t * h * w // 4 for t, h, w in hf_grids)
        if image_tokens != expected_image_tokens:
            raise AssertionError(
                f"sample={index} image token count differs: actual={image_tokens}, expected={expected_image_tokens}"
            )
        results.append(
            {
                "index": index,
                "qid": sample["extra_info"].get("qid"),
                "images": len(images),
                "prompt_tokens": len(hf_ids),
                "image_tokens": image_tokens,
                "image_grid_thw": hf_grids,
            }
        )

    output = {"samples": len(results), "architecture": architecture, "results": results}
    if args.result_path:
        path = Path(args.result_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print("OPENPANGU_PROCESSOR_PARITY_PASS " + json.dumps(output, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
