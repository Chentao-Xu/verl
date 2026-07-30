#!/usr/bin/env bash
# Formal 400-step convergence run:
# OpenPangu-VL-7B + ViRL39K + GRPO + FSDP2/vLLM on 8 Ascend NPUs.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-400}
export MODEL_PATH=${MODEL_PATH:-/home/chentaoxu/data/models/openPangu-VL-7B}
export TRAIN_FILE=${TRAIN_FILE:-/home/chentaoxu/data/datasets/virl39k/processed/train_single_image.parquet}
export TEST_FILE=${TEST_FILE:-/home/chentaoxu/data/datasets/virl39k/processed/val_single_image.parquet}

export NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-8}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
export ROLLOUT_N=${ROLLOUT_N:-4}

# The HF processor and vLLM do not count OpenPangu image tokens identically.
# Keep extra rollout context headroom for prompts accepted at the 3072 limit.
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-3072}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}
export MAX_MODEL_LENGTH=${MAX_MODEL_LENGTH:-8192}
export PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-4096}
export FILTER_OVERLONG_PROMPTS=${FILTER_OVERLONG_PROMPTS:-True}

export TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:--1}
export VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-256}
export TEST_FREQ=${TEST_FREQ:-10}
export SAVE_FREQ=${SAVE_FREQ:-50}

export TRAINER_LOGGER=${TRAINER_LOGGER:-'["console","wandb"]'}
export PROJECT_NAME=${PROJECT_NAME:-openpangu_vl_grpo_npu}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-openpangu_vl_7b_virl39k_grpo_400step_convergence}
export LOG_DIR=${LOG_DIR:-/home/chentaoxu/data/openpangu-vl-npu/convergence-400}
export CHECKPOINT_DIR=${CHECKPOINT_DIR:-/home/chentaoxu/data/openpangu-vl-npu/checkpoints/${EXPERIMENT_NAME}}
export VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-${LOG_DIR}/generations}
export RESUME_MODE=${RESUME_MODE:-auto}

export VERL_ROLLOUT_LOGPROB_CHECK=${VERL_ROLLOUT_LOGPROB_CHECK:-1}

CHECK_NPU_IDLE=${CHECK_NPU_IDLE:-1}
MIN_FREE_DISK_GB=${MIN_FREE_DISK_GB:-1000}

for required_path in "${MODEL_PATH}" "${TRAIN_FILE}" "${TEST_FILE}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Required path does not exist: ${required_path}" >&2
        exit 2
    fi
done

if [[ "${WANDB_MODE:-online}" == "online" ]] && ! command -v wandb >/dev/null 2>&1; then
    echo "wandb CLI is unavailable in this environment" >&2
    exit 3
fi

if [[ "${CHECK_NPU_IDLE}" == "1" ]]; then
    if ! command -v npu-smi >/dev/null 2>&1; then
        echo "npu-smi is unavailable; set CHECK_NPU_IDLE=0 to bypass this check" >&2
        exit 4
    fi
    idle_npu_count="$(npu-smi info | grep -c "No running processes found in NPU" || true)"
    if [[ "${idle_npu_count}" -ne "${NDEVICES_PER_NODE}" ]]; then
        echo "Expected ${NDEVICES_PER_NODE} idle NPUs, but found ${idle_npu_count}; refusing to start" >&2
        npu-smi info >&2
        exit 5
    fi
fi

mkdir -p "${LOG_DIR}" "${CHECKPOINT_DIR}" "${VALIDATION_DATA_DIR}"
available_disk_gb="$(df -Pk "${CHECKPOINT_DIR}" | awk 'NR == 2 {print int($4 / 1024 / 1024)}')"
if ((available_disk_gb < MIN_FREE_DISK_GB)); then
    echo "Only ${available_disk_gb} GiB is free; at least ${MIN_FREE_DISK_GB} GiB is required" >&2
    exit 6
fi

exec "${SCRIPT_DIR}/run_openpangu_vl_7b_npu.sh" \
    data.shuffle=True \
    data.seed=42 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    "$@"
