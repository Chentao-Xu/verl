#!/usr/bin/env bash
# 100-step OpenPangu-VL-7B GRPO acceptance run with W&B and checkpoints.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-100}
export ROLLOUT_N=${ROLLOUT_N:-4}
export MODEL_PATH=${MODEL_PATH:-/home/chentaoxu/data/models/openPangu-VL-7B}
export TRAIN_FILE=${TRAIN_FILE:-/home/chentaoxu/data/datasets/virl39k/processed/train_single_image.parquet}
export TEST_FILE=${TEST_FILE:-/home/chentaoxu/data/datasets/virl39k/processed/val_single_image.parquet}
export VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-256}
export TRAINER_LOGGER=${TRAINER_LOGGER:-'["console","wandb"]'}
export PROJECT_NAME=${PROJECT_NAME:-openpangu_vl_grpo_npu}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-openpangu_vl_7b_virl39k_grpo_100step_val256_wandb}
export LOG_DIR=${LOG_DIR:-/home/chentaoxu/data/openpangu-vl-npu/stage8}
export CHECKPOINT_DIR=${CHECKPOINT_DIR:-/home/chentaoxu/data/openpangu-vl-npu/checkpoints/${EXPERIMENT_NAME}}
export SAVE_FREQ=${SAVE_FREQ:-50}
export TEST_FREQ=${TEST_FREQ:-5}
export RESUME_MODE=${RESUME_MODE:-disable}
export VERL_ROLLOUT_LOGPROB_CHECK=${VERL_ROLLOUT_LOGPROB_CHECK:-1}

if [[ "${WANDB_MODE:-online}" == "online" ]] && ! command -v wandb >/dev/null 2>&1; then
    echo "wandb CLI is unavailable in this environment" >&2
    exit 3
fi

exec "${SCRIPT_DIR}/run_openpangu_vl_7b_npu.sh" "$@"
