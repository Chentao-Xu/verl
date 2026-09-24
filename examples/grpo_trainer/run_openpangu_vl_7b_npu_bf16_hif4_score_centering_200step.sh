#!/usr/bin/env bash
# BF16 Actor + simulated HiF4/HiF8 W4A8 rollout, standalone score centering.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
ENTRY="${SCRIPT_DIR}/run_openpangu_vl_7b_npu_bf16_hif4_w4a8_smoke.sh"
DATA=/home/chentaoxu/data/datasets/virl39k/processed/subsets/virl4k_seed42_v1
ROOT=/home/chentaoxu/data/openpangu-eval-work/training/virl4k-bf16-hif4-score-centering-20260925
EXPERIMENT=openpangu_vl_7b_virl4k_bf16_hif4_w4a8_sc128_b64n8_s200_seed42_accel_dynbsz4096_conc32
RUN="${ROOT}/${EXPERIMENT}"

if [[ -e "${RUN}/run_manifest.txt" ]] || [[ -e "${RUN}/checkpoints" ]]; then
    echo "Refusing to reuse existing experiment directory: ${RUN}" >&2
    exit 2
fi
mkdir -p "${RUN}" "${RUN}/generations"
trap 'status=$?; printf "exit_code=%s\nfinished_at=%s\n" "$status" "$(date -Is)" > "${RUN}/exit_status.txt"' EXIT

export MODEL_PATH=/home/chentaoxu/data/models/openPangu-VL-7B
export TRAIN_FILE="${DATA}/train_single_image_max3072_seed42_4000.parquet"
export TEST_FILE="${DATA}/test_single_image_fixed1698.parquet"
export TRAIN_MAX_SAMPLES=-1 VAL_MAX_SAMPLES=-1 FILTER_OVERLONG_PROMPTS=False
export TOTAL_TRAINING_STEPS=200 TRAIN_BATCH_SIZE=64 PPO_MINI_BATCH_SIZE=64 ROLLOUT_N=8
export MAX_PROMPT_LENGTH=7168 MAX_RESPONSE_LENGTH=1024 MAX_MODEL_LENGTH=16384
export PPO_MAX_TOKEN_LEN_PER_GPU=4096 ROLLOUT_GPU_MEM_UTIL=0.60 AGENT_NUM_WORKERS=8
export TEST_FREQ=25 SAVE_FREQ=100 LOG_VAL_GENERATIONS=16 RESUME_MODE=disable
export TRAINER_LOGGER='["console","wandb"]' WANDB_MODE=online
export PROJECT_NAME=openpangu_vl_virl4k_grpo EXPERIMENT_NAME="${EXPERIMENT}"
export LOG_DIR="${RUN}" CHECKPOINT_DIR="${RUN}/checkpoints"
export VALIDATION_DATA_DIR="${RUN}/generations" RAY_TMPDIR=/tmp/r_hif4_sc200
export HCCL_HOST_SOCKET_PORT_RANGE=62000-62050 HCCL_NPU_SOCKET_PORT_RANGE=63000-63050
export PYTHONPATH="/vllm:${PYTHONPATH:-}"

{
    printf 'started_at=%s\n' "$(date -Is)"
    printf 'hostname=%s\n' "$(hostname)"
    printf 'git_head=%s\n' "$(git -c safe.directory="${REPO}" -C "${REPO}" rev-parse HEAD)"
    printf 'dataset=virl4k_seed42_v1\nprecision=bf16_hif4_w4a8\n'
    printf 'train_rows=4000\neffective_rows_per_epoch=3968\nsteps_per_epoch=62\n'
    printf 'total_steps=200\ntrain_batch_size=64\nppo_mini_batch_size=64\nrollout_n=8\n'
    printf 'score_centering_topk=128\nrollout_correction=none\nactor_qat=false\n'
    printf 'test_freq=25\nval_max_samples=-1\nsave_freq=100\n'
    printf 'response_length=1024\nprompt_length=7168\n'
    printf 'dynamic_batch=true\ndynamic_token_budget=4096\nrollout_concurrency=32\n'
    printf 'actor_precision=bf16\nrollout_weight=hifx4\nrollout_activation=dhif8\n'
    printf 'low_precision_scope=language_only_layers_0_33\nvision_precision=bf16\n'
    printf 'train_sha256=%s\n' "$(sha256sum "${TRAIN_FILE}" | awk '{print $1}')"
    printf 'test_sha256=%s\n' "$(sha256sum "${TEST_FILE}" | awk '{print $1}')"
} > "${RUN}/run_manifest.txt"

bash "${ENTRY}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4096 \
    actor_rollout_ref.rollout.max_num_seqs=32 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.60 \
    actor_rollout_ref.rollout.score_centering_topk=128 \
    actor_rollout_ref.actor.policy_loss.loss_mode=score_centering \
    algorithm.rollout_correction.rollout_is=null \
    algorithm.rollout_correction.rollout_rs=null \
    trainer.total_epochs=6 \
    trainer.max_actor_ckpt_to_keep=10
