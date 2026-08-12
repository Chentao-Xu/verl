# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Two-phase OpenPangu native-vLLM versus HF-actor log-prob parity test.

Run the rollout phase first and let that process exit before starting the actor
phase. This keeps only one 7B model resident on the selected NPU at a time.
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

# Ascend cannot be initialized in a process forked after Torch import.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
from omegaconf import OmegaConf

from verl.utils.dataset.rl_dataset import RLHFDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("rollout", "rollout-prefill", "actor"))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--fixture-path", required=True)
    parser.add_argument("--result-path")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--actor-attention", choices=("eager", "sdpa", "fused-infer"), default="eager")
    parser.add_argument("--actor-position-ids", choices=("processor", "model-default"), default="processor")
    parser.add_argument("--actor-mxfp8", choices=("none", "weight_only", "w8a8"), default="none")
    parser.add_argument(
        "--actor-mxfp8-layer-range",
        help="Optional inclusive language layer range START:END; visual layers remain BF16.",
    )
    parser.add_argument("--rollout-mxfp8", choices=("none", "weight_only", "w8a8"), default="none")
    parser.add_argument(
        "--rollout-mxfp8-scope",
        choices=("all", "language_only", "visual_only"),
        default="all",
    )
    parser.add_argument("--rollout-mxfp8-layer-range", help="Optional inclusive language layer range START:END.")
    parser.add_argument("--mean-threshold", type=float, default=0.01)
    parser.add_argument("--p95-threshold", type=float, default=0.05)
    parser.add_argument("--max-threshold", type=float, default=0.15)
    parser.add_argument("--pearson-threshold", type=float, default=0.995)
    return parser.parse_args()


def dataset_config() -> OmegaConf:
    return OmegaConf.create(
        {
            "prompt_key": "prompt",
            "image_key": "images",
            "video_key": "videos",
            "audio_key": "audios",
            "max_prompt_length": 1536,
            "filter_overlong_prompts": False,
            "truncation": "error",
            "shuffle": False,
        }
    )


def load_sample(dataset_path: str, processor, index: int, text_only: bool = False):
    if text_only:
        messages = [{"role": "user", "content": [{"type": "text", "text": "请用三句话介绍强化学习。"}]}]
        rendered = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return {"extra_info": {"qid": "synthetic-text-only"}}, rendered, []
    dataset = RLHFDataset(dataset_path, processor.tokenizer, dataset_config(), processor=processor)
    sample = dataset[index]
    messages = sample["raw_prompt"]
    rendered = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    images, videos, audios = dataset._process_multi_modal_info(
        messages, image_patch_size=dataset.image_patch_size, config=dataset.config
    )
    if videos or audios or len(images) != 1:
        raise AssertionError("The log-prob parity fixture must contain exactly one image")
    return sample, rendered, images


def run_rollout(args: argparse.Namespace) -> None:
    # Python loader monkey patches do not propagate into the spawned V1
    # EngineCore process. Keep the tiny parity fixture in-process whenever the
    # MXFP8 simulator is requested so the patched loader is the one exercised.
    if args.rollout_mxfp8 != "none":
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    from vllm import LLM, SamplingParams

    from verl.utils import hf_processor

    if args.rollout_mxfp8 != "none":
        # vLLM's logprob rank helper is torch.compile-decorated and currently
        # probes CUDA capability even on NPU in this in-process test mode.
        # Replace only this test helper with its original eager callable.
        from vllm.v1.sample import sampler as sampler_module
        from vllm.v1.sample.ops import logprobs as logprobs_ops

        eager_count = logprobs_ops.batched_count_greater_than.__wrapped__
        logprobs_ops.batched_count_greater_than = eager_count
        sampler_module.batched_count_greater_than = eager_count
        os.environ["VERL_VLLM_MXFP8_SIM_ENABLED"] = "1"
        os.environ["VERL_MXFP8_SIM_MODE"] = args.rollout_mxfp8
        os.environ["VERL_MXFP8_SIM_SCOPE"] = args.rollout_mxfp8_scope
        os.environ["VERL_MXFP8_SIM_LANGUAGE_LAYER_RANGE"] = args.rollout_mxfp8_layer_range or ""
        from verl.utils.vllm.mxfp8_sim import apply_mxfp8_sim_patches

        apply_mxfp8_sim_patches()

    processor = hf_processor(args.model_path, trust_remote_code=True, use_fast=True)
    sample, rendered, images = load_sample(args.dataset_path, processor, args.sample_index, args.text_only)
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        max_num_seqs=1,
        enforce_eager=True,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        limit_mm_per_prompt={"image": 1, "video": 0},
        mm_processor_cache_gb=0,
        gpu_memory_utilization=0.45,
        seed=0,
    )
    outputs = llm.generate(
        [rendered if args.text_only else {"prompt": rendered, "multi_modal_data": {"image": images[0]}}],
        SamplingParams(temperature=0.0, max_tokens=args.max_tokens, logprobs=1, seed=0),
        use_tqdm=False,
    )
    request_output = outputs[0]
    candidate = request_output.outputs[0]
    rollout_log_probs = []
    for token_id, token_log_probs in zip(candidate.token_ids, candidate.logprobs, strict=True):
        if token_id not in token_log_probs:
            raise AssertionError(f"vLLM did not return the sampled token log-prob for token {token_id}")
        rollout_log_probs.append(float(token_log_probs[token_id].logprob))
    fixture = {
        "sample_index": args.sample_index,
        "text_only": args.text_only,
        "qid": sample["extra_info"].get("qid"),
        "prompt_token_ids": list(request_output.prompt_token_ids),
        "response_token_ids": list(candidate.token_ids),
        "rollout_log_probs": rollout_log_probs,
        "text": candidate.text,
        "finish_reason": candidate.finish_reason,
        "rollout_mxfp8": args.rollout_mxfp8,
        "rollout_mxfp8_scope": args.rollout_mxfp8_scope,
        "rollout_mxfp8_layer_range": args.rollout_mxfp8_layer_range,
    }
    path = Path(args.fixture_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fixture, ensure_ascii=False, indent=2) + "\n")
    print("OPENPANGU_ROLLOUT_LOGPROB_CAPTURE_PASS " + json.dumps(fixture, ensure_ascii=False), flush=True)


def run_rollout_prefill(args: argparse.Namespace) -> None:
    from vllm import LLM, SamplingParams

    fixture_path = Path(args.fixture_path)
    fixture = json.loads(fixture_path.read_text())
    if not fixture.get("text_only"):
        raise AssertionError("rollout-prefill currently supports text-only fixtures")
    combined_ids = fixture["prompt_token_ids"] + fixture["response_token_ids"]
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        max_num_seqs=1,
        enforce_eager=True,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        limit_mm_per_prompt={"image": 1, "video": 0},
        mm_processor_cache_gb=0,
        gpu_memory_utilization=0.45,
        seed=0,
    )
    output = llm.generate(
        [{"prompt_token_ids": combined_ids}],
        SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1, seed=0),
        use_tqdm=False,
    )[0]
    prompt_log_probs = output.prompt_logprobs
    prompt_length = len(fixture["prompt_token_ids"])
    prefill_log_probs = []
    for position, token_id in enumerate(fixture["response_token_ids"], start=prompt_length):
        token_log_probs = prompt_log_probs[position]
        if token_log_probs is None or token_id not in token_log_probs:
            raise AssertionError(f"Missing prompt log-prob at position={position}, token={token_id}")
        prefill_log_probs.append(float(token_log_probs[token_id].logprob))
    fixture["rollout_prefill_log_probs"] = prefill_log_probs
    fixture_path.write_text(json.dumps(fixture, ensure_ascii=False, indent=2) + "\n")
    print("OPENPANGU_ROLLOUT_PREFILL_LOGPROB_CAPTURE_PASS " + json.dumps(prefill_log_probs), flush=True)


def extend_sequence_inputs(inputs: dict, response_ids: torch.Tensor) -> None:
    response_length = response_ids.shape[-1]
    inputs["input_ids"] = torch.cat((inputs["input_ids"], response_ids), dim=-1)
    inputs["attention_mask"] = torch.cat(
        (inputs["attention_mask"], torch.ones_like(response_ids, dtype=inputs["attention_mask"].dtype)), dim=-1
    )
    for key in ("mm_token_type_ids", "token_type_ids"):
        value = inputs.get(key)
        if isinstance(value, torch.Tensor) and value.ndim == 2:
            inputs[key] = torch.cat((value, torch.zeros_like(response_ids, dtype=value.dtype)), dim=-1)
    if response_length == 0:
        raise AssertionError("vLLM generated no response tokens")


def add_position_ids(processor, inputs: dict) -> None:
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


def run_actor(args: argparse.Namespace) -> None:
    import torch_npu  # noqa: F401

    from verl.models.transformers.monkey_patch import apply_monkey_patch
    from verl.utils.model import get_hf_auto_model_class
    from verl.workers.config.model import HFModelConfig

    fixture = json.loads(Path(args.fixture_path).read_text())
    torch.npu.set_device(args.device)
    model_config = HFModelConfig(
        path=args.model_path,
        trust_remote_code=True,
        override_config={"attn_implementation": "sdpa" if args.actor_attention == "sdpa" else "eager"},
    )
    processor = model_config.processor
    _, rendered, images = load_sample(
        args.dataset_path, processor, fixture["sample_index"], fixture.get("text_only", False)
    )
    inputs = dict(processor(text=[rendered], images=images or None, padding=False, return_tensors="pt"))
    prompt_ids = inputs["input_ids"][0].tolist()
    if prompt_ids != fixture["prompt_token_ids"]:
        mismatch = next(
            (i for i, pair in enumerate(zip(prompt_ids, fixture["prompt_token_ids"])) if pair[0] != pair[1]), None
        )
        raise AssertionError(f"HF/vLLM prompt token mismatch at {mismatch}")

    response_ids = torch.tensor([fixture["response_token_ids"]], dtype=torch.long)
    prompt_length = inputs["input_ids"].shape[-1]
    extend_sequence_inputs(inputs, response_ids)
    if args.actor_position_ids == "processor":
        add_position_ids(processor, inputs)

    auto_class = get_hf_auto_model_class(model_config.hf_config)
    model, loading_info = auto_class.from_pretrained(
        model_config.local_path,
        config=model_config.hf_config,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        output_loading_info=True,
    )
    if any(loading_info[key] for key in ("missing_keys", "unexpected_keys", "mismatched_keys")):
        raise AssertionError(f"Weight loading issues: {loading_info}")
    apply_monkey_patch(model=model, use_remove_padding=False)
    if args.actor_mxfp8 != "none":
        from verl.utils.qat.core import apply_qat

        qat_mode = "mxfp8_w8a16" if args.actor_mxfp8 == "weight_only" else "mxfp8_w8a8"
        ignore_patterns = ["lm_head", "embed_tokens"]
        if args.actor_mxfp8_layer_range:
            try:
                start_text, end_text = args.actor_mxfp8_layer_range.split(":", maxsplit=1)
                start, end = int(start_text), int(end_text)
            except ValueError as exc:
                raise ValueError("--actor-mxfp8-layer-range must be START:END") from exc
            if start < 0 or end < start:
                raise ValueError("--actor-mxfp8-layer-range must satisfy 0 <= START <= END")
            layers = "|".join(str(layer) for layer in range(start, end + 1))
            ignore_patterns.extend(
                [
                    "re:^(?!model\\.language_model\\.layers\\.(?:" + layers + ")\\.)",
                    "visual",
                ]
            )
        model = apply_qat(
            model,
            {
                "enable": True,
                "mode": qat_mode,
                "group_size": 32,
                "ignore_patterns": ignore_patterns,
                "activation_observer": "static_minmax",
            },
        )
    if args.actor_attention == "fused-infer":
        for module in model.modules():
            if module.__class__.__name__ in {"PanguEmbeddedAttention", "OpenPanguVLVisionAttention"}:
                sys.modules[module.__class__.__module__].NPU_ATTN_INFR = True
    model = model.to(args.device).eval()
    inputs = {key: value.to(args.device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs, use_cache=False).logits.float()
    response_logits = logits[:, prompt_length - 1 : -1]
    actor_log_probs = torch.log_softmax(response_logits, dim=-1).gather(
        -1, response_ids.to(args.device).unsqueeze(-1)
    ).squeeze(-1)[0].cpu()
    rollout_log_probs = torch.tensor(fixture["rollout_log_probs"], dtype=torch.float32)
    diff = (actor_log_probs - rollout_log_probs).abs()
    metrics = {
        "tokens": diff.numel(),
        "actor_attention": args.actor_attention,
        "actor_position_ids": args.actor_position_ids,
        "actor_mxfp8": args.actor_mxfp8,
        "actor_mxfp8_layer_range": args.actor_mxfp8_layer_range,
        "mean_abs_logprob_diff": diff.mean().item(),
        "p95_abs_logprob_diff": torch.quantile(diff, 0.95).item(),
        "max_abs_logprob_diff": diff.max().item(),
        "pearson": torch.corrcoef(torch.stack((actor_log_probs, rollout_log_probs)))[0, 1].item(),
        "actor_log_probs": actor_log_probs.tolist(),
        "rollout_log_probs": rollout_log_probs.tolist(),
    }
    if "rollout_prefill_log_probs" in fixture:
        prefill_log_probs = torch.tensor(fixture["rollout_prefill_log_probs"], dtype=torch.float32)
        prefill_diff = (actor_log_probs - prefill_log_probs).abs()
        metrics["prefill"] = {
            "mean_abs_logprob_diff": prefill_diff.mean().item(),
            "p95_abs_logprob_diff": torch.quantile(prefill_diff, 0.95).item(),
            "max_abs_logprob_diff": prefill_diff.max().item(),
            "pearson": torch.corrcoef(torch.stack((actor_log_probs, prefill_log_probs)))[0, 1].item(),
            "rollout_prefill_log_probs": prefill_log_probs.tolist(),
        }
    if args.result_path:
        result_path = Path(args.result_path)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print("OPENPANGU_LOGPROB_PARITY_RESULT " + json.dumps(metrics), flush=True)
    failures = []
    for name, threshold in (
        ("mean_abs_logprob_diff", args.mean_threshold),
        ("p95_abs_logprob_diff", args.p95_threshold),
        ("max_abs_logprob_diff", args.max_threshold),
    ):
        if metrics[name] > threshold:
            failures.append(f"{name}={metrics[name]:.6f}>{threshold:.6f}")
    if metrics["pearson"] < args.pearson_threshold:
        failures.append(f"pearson={metrics['pearson']:.6f}<{args.pearson_threshold:.6f}")

    del logits, model, inputs
    gc.collect()
    torch.npu.empty_cache()
    if failures:
        raise AssertionError("OpenPangu log-prob parity failed: " + ", ".join(failures))
    print("OPENPANGU_LOGPROB_PARITY_PASS", flush=True)


def main() -> None:
    args = parse_args()
    if args.phase == "rollout":
        run_rollout(args)
    elif args.phase == "rollout-prefill":
        run_rollout_prefill(args)
    else:
        run_actor(args)


if __name__ == "__main__":
    main()
