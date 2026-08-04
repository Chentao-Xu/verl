#!/usr/bin/env bash
# 20-step acceptance run for OpenPangu-VL-7B rollout correction.
#
# Keep the proven actor attention backend (SDPA) and the convergence run's
# remaining GRPO settings fixed. The only algorithmic change under test is
# decoupled token-level truncated importance sampling (Token-TIS).

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-20}
export MODEL_PATH=${MODEL_PATH:-/home/chentaoxu/data/models/openPangu-VL-7B}
export TRAIN_FILE=${TRAIN_FILE:-/home/chentaoxu/data/datasets/virl39k/processed/train_single_image_max3072.parquet}
export TEST_FILE=${TEST_FILE:-/home/chentaoxu/data/datasets/virl39k/processed/val_single_image_seed42_raw256_max3072.parquet}

export NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-8}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
export ROLLOUT_N=${ROLLOUT_N:-4}

export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-3072}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}
export MAX_MODEL_LENGTH=${MAX_MODEL_LENGTH:-8192}
export PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-4096}
export FILTER_OVERLONG_PROMPTS=${FILTER_OVERLONG_PROMPTS:-False}
# Restore the original actor operator explicitly; do not inherit fusion settings.
export ACTOR_ATTN_IMPLEMENTATION=${ACTOR_ATTN_IMPLEMENTATION:-sdpa}

export TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:--1}
# Fixed file: 256 seed-42 raw rows, with one overlong sample removed in advance.
export VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:--1}
export TEST_FREQ=${TEST_FREQ:-10}
export SAVE_FREQ=${SAVE_FREQ:-20}

export TRAINER_LOGGER=${TRAINER_LOGGER:-'["console","wandb"]'}
export PROJECT_NAME=${PROJECT_NAME:-openpangu_vl_grpo_npu}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-openpangu_vl_7b_virl39k_grpo_20step_sdpa_token_tis}
export LOG_DIR=${LOG_DIR:-/home/chentaoxu/data/openpangu-vl-npu/sdpa-token-tis-20step}
export CHECKPOINT_DIR=${CHECKPOINT_DIR:-/home/chentaoxu/data/openpangu-vl-npu/checkpoints/${EXPERIMENT_NAME}}
export VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-${LOG_DIR}/generations}
export RESUME_MODE=${RESUME_MODE:-disable}

export VERL_ROLLOUT_LOGPROB_CHECK=${VERL_ROLLOUT_LOGPROB_CHECK:-1}
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/r_tis20}
export HCCL_HOST_SOCKET_PORT_RANGE=${HCCL_HOST_SOCKET_PORT_RANGE:-64000-64050}
export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-65000-65050}

REQUIRED_BASE_COMMIT=${REQUIRED_BASE_COMMIT:-6ad023956430a33512e09baf9e3c01163b5f420e}
EXPECTED_TRAIN_SHA256=${EXPECTED_TRAIN_SHA256:-4b4f4216922c576a306f7b4d78b4dc2b99463c83f618f10dfdf4b25eae44d352}
EXPECTED_VAL_SHA256=${EXPECTED_VAL_SHA256:-001a75031eebb79bd88cb4ff30d53b3ca9cd1732a7f5b815da670843c73e803d}
CHECK_NPU_IDLE=${CHECK_NPU_IDLE:-1}
MIN_FREE_DISK_GB=${MIN_FREE_DISK_GB:-200}

for required_path in "${MODEL_PATH}" "${TRAIN_FILE}" "${TEST_FILE}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Required path does not exist: ${required_path}" >&2
        exit 2
    fi
done

actual_commit="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
if ! git -C "${REPO_ROOT}" merge-base --is-ancestor "${REQUIRED_BASE_COMMIT}" "${actual_commit}"; then
    echo "Required base commit ${REQUIRED_BASE_COMMIT} is not an ancestor of ${actual_commit}" >&2
    exit 3
fi

if [[ -n "$(git -C "${REPO_ROOT}" status --short --untracked-files=no)" ]]; then
    echo "Tracked files are modified; refusing to start a non-reproducible run" >&2
    git -C "${REPO_ROOT}" status --short --untracked-files=no >&2
    exit 4
fi

echo "Experiment git HEAD: ${actual_commit}"
echo "Required base commit: ${REQUIRED_BASE_COMMIT}"

printf '%s  %s\n' "${EXPECTED_TRAIN_SHA256}" "${TRAIN_FILE}" | sha256sum --check --status
printf '%s  %s\n' "${EXPECTED_VAL_SHA256}" "${TEST_FILE}" | sha256sum --check --status

if [[ "${WANDB_MODE:-online}" == "online" ]] && [[ "${TRAINER_LOGGER}" == *wandb* ]]; then
    if ! command -v wandb >/dev/null 2>&1; then
        echo "wandb CLI is unavailable in this environment" >&2
        exit 5
    fi
    if ! wandb login >/dev/null 2>&1; then
        echo "wandb credentials are unavailable; run 'wandb login' in the training container" >&2
        exit 6
    fi
fi

if [[ "${CHECK_NPU_IDLE}" == "1" ]]; then
    if ! command -v npu-smi >/dev/null 2>&1; then
        echo "npu-smi is unavailable; set CHECK_NPU_IDLE=0 to bypass this check" >&2
        exit 7
    fi
    idle_npu_count="$(npu-smi info | grep -c "No running processes found in NPU" || true)"
    if [[ "${idle_npu_count}" -ne "${NDEVICES_PER_NODE}" ]]; then
        echo "Expected ${NDEVICES_PER_NODE} idle NPUs, but found ${idle_npu_count}; refusing to start" >&2
        npu-smi info >&2
        exit 8
    fi
fi

mkdir -p "${RAY_TMPDIR}" "${LOG_DIR}" "${CHECKPOINT_DIR}" "${VALIDATION_DATA_DIR}"
available_disk_gb="$(df -Pk "${CHECKPOINT_DIR}" | awk 'NR == 2 {print int($4 / 1024 / 1024)}')"
if ((available_disk_gb < MIN_FREE_DISK_GB)); then
    echo "Only ${available_disk_gb} GiB is free; at least ${MIN_FREE_DISK_GB} GiB is required" >&2
    exit 9
fi

if [[ "${PREPARE_ONLY:-0}" == "1" ]]; then
    echo "Preflight passed; training was not started"
    exit 0
fi

exec "${SCRIPT_DIR}/run_openpangu_vl_7b_npu.sh" \
    data.shuffle=True \
    data.seed=42 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    algorithm.rollout_correction.rollout_is_batch_normalize=False \
    algorithm.rollout_correction.rollout_rs=null \
    algorithm.rollout_correction.rollout_rs_threshold=null \
    algorithm.rollout_correction.bypass_mode=False \
    "$@"
