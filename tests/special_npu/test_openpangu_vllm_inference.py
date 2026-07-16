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

"""OpenPangu-VL vLLM-Ascend text, image, concurrency, and stability test."""

import argparse
import io
import json
import os
import time
from pathlib import Path

# Forking an initialized OpenMP pool crashes the ARM PyTorch build used by the
# Ascend image. These defaults must be set before importing vLLM and torch.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import pandas as pd
from PIL import Image
from vllm import LLM, SamplingParams, __version__ as vllm_version

SYSTEM_PROMPT = (
    "You are a multimodal large model developed by Huawei, named openPangu-VL-7B. "
    "You can process both text and visual inputs and generate text outputs."
)
GEO3K_INDICES = (0, 266, 460, 29, 378, 164, 203, 141, 251, 313)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--geo3k-path", default="/data/datasets/geo3k/test.parquet")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--suite", choices=("basic", "full"), default="basic")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--stress-requests", type=int, default=50)
    parser.add_argument("--result-path")
    return parser.parse_args()


def format_prompt(question: str, include_image: bool) -> str:
    question = question.replace("<image>", "").strip()
    image_placeholder = "[unused18][unused19][unused20]" if include_image else ""
    return (
        f"<s>[unused9]系统：{SYSTEM_PROMPT}[unused10]"
        f"[unused9]用户：{image_placeholder}{question}[unused10]"
        "[unused9]助手："
    )


def load_geo3k_samples(path: str) -> list[dict]:
    frame = pd.read_parquet(path)
    samples = []
    for index in GEO3K_INDICES:
        row = frame.iloc[index]
        image = Image.open(io.BytesIO(row.images[0]["bytes"])).convert("RGB")
        samples.append(
            {
                "index": int(index),
                "image": image,
                "question": row.prompt[0]["content"],
                "ground_truth": row.reward_model["ground_truth"],
            }
        )
    return samples


def request_from_sample(sample: dict, suffix: str = "") -> dict:
    question = sample["question"]
    if suffix:
        question = f"{question}\n{suffix}"
    return {
        "prompt": format_prompt(question, include_image=True),
        "multi_modal_data": {"image": sample["image"]},
    }


def summarize_output(output, *, name: str, metadata: dict | None = None) -> dict:
    candidate = output.outputs[0]
    if not candidate.token_ids:
        raise AssertionError(f"{name} produced no output tokens")
    if not candidate.text.strip():
        raise AssertionError(f"{name} produced only whitespace")
    result = {
        "name": name,
        "text": candidate.text,
        "token_count": len(candidate.token_ids),
        "finish_reason": candidate.finish_reason,
    }
    if metadata:
        result.update(metadata)
    return result


def generate(llm: LLM, requests: list[dict | str], *, max_tokens: int) -> tuple[list, float]:
    start = time.monotonic()
    outputs = llm.generate(
        requests,
        sampling_params=SamplingParams(temperature=0.0, max_tokens=max_tokens),
        use_tqdm=False,
    )
    return outputs, time.monotonic() - start


def run_basic(llm: LLM, samples: list[dict], max_tokens: int) -> list[dict]:
    text_prompt = format_prompt("请用一句话介绍你自己。", include_image=False)
    text_outputs, text_seconds = generate(llm, [text_prompt], max_tokens=max_tokens)
    text_result = summarize_output(text_outputs[0], name="tp_text")
    text_result["elapsed_seconds"] = text_seconds
    print("OPENPANGU_TP_TEXT_PASS " + json.dumps(text_result, ensure_ascii=False), flush=True)

    sample = samples[0]
    image_outputs, image_seconds = generate(llm, [request_from_sample(sample)], max_tokens=max_tokens)
    image_result = summarize_output(
        image_outputs[0],
        name="tp_image",
        metadata={
            "geo3k_index": sample["index"],
            "image_size": list(sample["image"].size),
            "ground_truth": sample["ground_truth"],
        },
    )
    image_result["elapsed_seconds"] = image_seconds
    print("OPENPANGU_TP_IMAGE_PASS " + json.dumps(image_result, ensure_ascii=False), flush=True)
    return [text_result, image_result]


def run_full(llm: LLM, samples: list[dict], stress_requests: int) -> dict:
    multires_requests = [request_from_sample(sample) for sample in samples]
    multires_outputs, multires_seconds = generate(llm, multires_requests, max_tokens=16)
    multires_results = [
        summarize_output(
            output,
            name=f"multires_{sample['index']}",
            metadata={"geo3k_index": sample["index"], "image_size": list(sample["image"].size)},
        )
        for output, sample in zip(multires_outputs, samples, strict=True)
    ]
    print(
        f"OPENPANGU_MULTIRES_PASS requests={len(multires_results)} elapsed_seconds={multires_seconds:.3f}",
        flush=True,
    )

    concurrent_requests = [request_from_sample(sample, suffix=f"并发请求 {i}") for i, sample in enumerate(samples[:8])]
    concurrent_outputs, concurrent_seconds = generate(llm, concurrent_requests, max_tokens=8)
    concurrent_results = [
        summarize_output(output, name=f"concurrent_{i}") for i, output in enumerate(concurrent_outputs)
    ]
    print(
        f"OPENPANGU_CONCURRENT_PASS requests={len(concurrent_results)} elapsed_seconds={concurrent_seconds:.3f}",
        flush=True,
    )

    stress_results = []
    stress_start = time.monotonic()
    for start in range(0, stress_requests, 8):
        batch_size = min(8, stress_requests - start)
        requests = [
            request_from_sample(samples[(start + offset) % len(samples)], suffix=f"稳定性请求 {start + offset}")
            for offset in range(batch_size)
        ]
        outputs, _ = generate(llm, requests, max_tokens=8)
        stress_results.extend(
            summarize_output(output, name=f"stress_{start + offset}") for offset, output in enumerate(outputs)
        )
    stress_seconds = time.monotonic() - stress_start
    if len(stress_results) != stress_requests:
        raise AssertionError(f"Stress request count mismatch: {len(stress_results)} != {stress_requests}")
    print(
        f"OPENPANGU_STRESS_PASS requests={len(stress_results)} elapsed_seconds={stress_seconds:.3f}",
        flush=True,
    )
    return {
        "multires": multires_results,
        "multires_elapsed_seconds": multires_seconds,
        "concurrent": concurrent_results,
        "concurrent_elapsed_seconds": concurrent_seconds,
        "stress_request_count": len(stress_results),
        "stress_elapsed_seconds": stress_seconds,
    }


def main() -> None:
    args = parse_args()
    samples = load_geo3k_samples(args.geo3k_path)
    init_start = time.monotonic()
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_num_seqs=8,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 1, "video": 0},
        mm_processor_cache_gb=0,
        gpu_memory_utilization=0.7,
        seed=0,
    )
    init_seconds = time.monotonic() - init_start
    print(
        f"OPENPANGU_ENGINE_READY tp={args.tensor_parallel_size} elapsed_seconds={init_seconds:.3f}",
        flush=True,
    )

    result = {
        "vllm_version": vllm_version,
        "model_path": args.model_path,
        "tensor_parallel_size": args.tensor_parallel_size,
        "suite": args.suite,
        "engine_init_seconds": init_seconds,
        "basic": run_basic(llm, samples, args.max_tokens),
    }
    if args.suite == "full":
        result["full"] = run_full(llm, samples, args.stress_requests)

    if args.result_path:
        result_path = Path(args.result_path)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"OPENPANGU_VLLM_INFERENCE_PASS tp={args.tensor_parallel_size} suite={args.suite}", flush=True)


if __name__ == "__main__":
    main()
