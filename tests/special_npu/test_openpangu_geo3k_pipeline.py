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

"""Validate one real Geo3K sample through OpenPangu's multimodal processor.

Run with:
    python tests/special_npu/test_openpangu_geo3k_pipeline.py \
        --model-path /data/models/openPangu-VL-7B \
        --dataset-path /data/datasets/geo3k/test.parquet
"""

import argparse
import io
import json
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf
from PIL import Image

from verl.utils import hf_processor
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.tokenizer import build_multimodal_processor_inputs, normalize_token_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--result-path")
    return parser.parse_args()


def inspect_source_row(dataset_path: str) -> dict:
    frame = pd.read_parquet(dataset_path)
    row = frame.iloc[0]
    required_columns = {"prompt", "images", "reward_model", "extra_info"}
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise AssertionError(f"Missing Geo3K columns: {sorted(missing_columns)}")
    if row.data_source != "hiyouga/geometry3k":
        raise AssertionError(f"Unexpected data source: {row.data_source}")
    if len(row.images) != 1 or not row.images[0].get("bytes"):
        raise AssertionError("The first Geo3K sample must contain one embedded image")
    if row.prompt[0]["content"].count("<image>") != 1:
        raise AssertionError("The first Geo3K prompt must contain exactly one <image> placeholder")

    image = Image.open(io.BytesIO(row.images[0]["bytes"]))
    image.load()
    ground_truth = row.reward_model["ground_truth"]
    if ground_truth != row.extra_info["answer"]:
        raise AssertionError("reward_model.ground_truth and extra_info.answer disagree")
    return {
        "row_count": len(frame),
        "columns": list(frame.columns),
        "image_format": image.format,
        "image_mode": image.mode,
        "image_size": list(image.size),
        "ground_truth": ground_truth,
    }


def run_processor_pipeline(model_path: str, dataset_path: str) -> dict:
    processor = hf_processor(model_path, trust_remote_code=True, use_fast=True)
    if processor is None:
        raise AssertionError("OpenPangu processor failed to initialize")

    config = OmegaConf.create(
        {
            "prompt_key": "prompt",
            "image_key": "images",
            "max_prompt_length": 1536,
            "filter_overlong_prompts": False,
            "shuffle": False,
            "trust_remote_code": True,
        }
    )
    dataset = RLHFDataset(
        data_files=dataset_path,
        tokenizer=processor.tokenizer,
        config=config,
        processor=processor,
        max_samples=1,
    )
    sample = dataset[0]
    messages = sample["raw_prompt"]
    content = messages[0]["content"]
    image_items = [item for item in content if item.get("type") == "image"]
    if len(image_items) != 1 or not isinstance(image_items[0].get("image"), Image.Image):
        raise AssertionError("RLHFDataset did not convert <image> into one PIL image item")

    raw_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    placeholder = processor.vision_start_token + processor.image_token + processor.vision_end_token
    if raw_prompt.count(placeholder) != 1:
        raise AssertionError("OpenPangu chat template did not emit exactly one image placeholder")

    images, videos, audios = dataset._process_multi_modal_info(
        messages,
        image_patch_size=dataset.image_patch_size,
        config=config,
    )
    model_inputs = build_multimodal_processor_inputs(
        processor,
        text=[raw_prompt],
        images=images,
        videos=videos,
        audio=audios,
    )
    prompt_ids = normalize_token_ids(model_inputs["input_ids"])
    image_grid_thw = model_inputs.get("image_grid_thw")
    if not prompt_ids or image_grid_thw is None or image_grid_thw.shape[0] != 1:
        raise AssertionError("OpenPangu processor did not produce valid text and image tensors")
    if len(prompt_ids) > config.max_prompt_length:
        raise AssertionError(f"Prompt has {len(prompt_ids)} tokens, over limit {config.max_prompt_length}")

    return {
        "processor": type(processor).__name__,
        "prompt_tokens": len(prompt_ids),
        "image_grid_thw": image_grid_thw.tolist(),
        "reward_model": sample["reward_model"],
        "extra_info": sample["extra_info"],
    }


def main() -> None:
    args = parse_args()
    result = {
        "source": inspect_source_row(args.dataset_path),
        "processor_pipeline": run_processor_pipeline(args.model_path, args.dataset_path),
    }
    if args.result_path:
        result_path = Path(args.result_path)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("OPENPANGU_GEO3K_PIPELINE_PASS " + json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
