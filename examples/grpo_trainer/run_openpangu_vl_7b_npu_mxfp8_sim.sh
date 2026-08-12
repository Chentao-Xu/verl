#!/usr/bin/env bash
# OpenPangu-VL-7B GRPO with an NPU MXFP8 quantize-dequantize rollout simulator.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

export VERL_MXFP8_SIM_MODE=${VERL_MXFP8_SIM_MODE:-weight_only}
if [[ "${VERL_MXFP8_SIM_MODE}" != "weight_only" && "${VERL_MXFP8_SIM_MODE}" != "w8a8" ]]; then
    echo "VERL_MXFP8_SIM_MODE must be weight_only or w8a8" >&2
    exit 2
fi

export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-1}
export MODEL_PATH=${MODEL_PATH:-/home/chentaoxu/data/models/openPangu-VL-7B}
export TRAIN_FILE=${TRAIN_FILE:-/home/chentaoxu/data/datasets/virl39k/processed/train_single_image_max3072.parquet}
export TEST_FILE=${TEST_FILE:-/home/chentaoxu/data/datasets/virl39k/processed/val_single_image_seed42_raw256_max3072.parquet}

export NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-8}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
export ROLLOUT_N=${ROLLOUT_N:-2}
export TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-16}
export VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-8}

export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-3072}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-64}
export MAX_MODEL_LENGTH=${MAX_MODEL_LENGTH:-4096}
export PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-4096}
export ACTOR_ATTN_IMPLEMENTATION=${ACTOR_ATTN_IMPLEMENTATION:-sdpa}

export TEST_FREQ=${TEST_FREQ:--1}
export SAVE_FREQ=${SAVE_FREQ:--1}
export LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS:-8}
export TRAINER_LOGGER=${TRAINER_LOGGER:-'["console"]'}
export PROJECT_NAME=${PROJECT_NAME:-openpangu_vl_grpo_npu}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-openpangu_vl_7b_mxfp8_sim_${VERL_MXFP8_SIM_MODE}_${TOTAL_TRAINING_STEPS}step}
export LOG_DIR=${LOG_DIR:-/home/chentaoxu/data/openpangu-vl-npu/mxfp8-sim-20260812/${EXPERIMENT_NAME}}
export CHECKPOINT_DIR=${CHECKPOINT_DIR:-${LOG_DIR}/checkpoints}
export VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-${LOG_DIR}/generations}
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/r_mxfp8_sim}

export PYTHONPATH="${REPO_ROOT}:/home/chentaoxu/quant_dequant:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/home/chentaoxu/quant_dequant/quant_cy_npu/cusrc:/usr/local/python3.11.15/lib/python3.11/site-packages/torch_npu/lib:/usr/local/python3.11.15/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export VERL_ROLLOUT_LOGPROB_CHECK=${VERL_ROLLOUT_LOGPROB_CHECK:-1}
export VERL_WEIGHT_SYNC_AUDIT=${VERL_WEIGHT_SYNC_AUDIT:-1}

idle_npu_count="$(npu-smi info | grep -c "No running processes found in NPU" || true)"
if [[ "${idle_npu_count}" -ne "${NDEVICES_PER_NODE}" ]]; then
    echo "Expected ${NDEVICES_PER_NODE} idle NPUs, found ${idle_npu_count}" >&2
    npu-smi info >&2
    exit 3
fi

mkdir -p "${LOG_DIR}" "${CHECKPOINT_DIR}" "${VALIDATION_DATA_DIR}" "${RAY_TMPDIR}"
git -C "${REPO_ROOT}" rev-parse HEAD | tee "${LOG_DIR}/git_head.txt"
env | grep -E '^(VERL_MXFP8_SIM_MODE|TOTAL_TRAINING_STEPS|TRAIN_BATCH_SIZE|ROLLOUT_N|MAX_RESPONSE_LENGTH)=' \
    | sort | tee "${LOG_DIR}/experiment_env.txt"

exec "${SCRIPT_DIR}/run_openpangu_vl_7b_npu.sh" \
    data.shuffle=True \
    data.seed=42 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.quantization=mxfp8_sim \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    algorithm.rollout_correction.rollout_is_batch_normalize=False \
    algorithm.rollout_correction.rollout_rs=null \
    algorithm.rollout_correction.rollout_rs_threshold=null \
    algorithm.rollout_correction.bypass_mode=False \
    "$@"
