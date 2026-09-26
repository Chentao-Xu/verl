#!/usr/bin/env bash
# Resume the completed 200-step BF16/HiF4 score-centering run to 15 epochs.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/run_openpangu_vl_7b_npu_bf16_hif4_w4a8_smoke.sh"
DATA=/home/chentaoxu/data/datasets/virl39k/processed/subsets/virl4k_seed42_v1
ROOT=/home/chentaoxu/data/openpangu-eval-work/training/virl4k-bf16-hif4-score-centering-20260925
EXPERIMENT=openpangu_vl_7b_virl4k_bf16_hif4_w4a8_sc128_b64n8_s200_seed42_accel_dynbsz4096_conc32
RUN="${ROOT}/${EXPERIMENT}"
CONT="${RUN}/continuation_step200_to930"
CKPT="${RUN}/checkpoints/global_step_200"

grep -qx 'exit_code=0' "${RUN}/exit_status.txt"
[[ -f "${CKPT}/data.pt" && -d "${CKPT}/actor" ]]
for rank in {0..7}; do
    for part in model optim extra_state; do
        [[ -s "${CKPT}/actor/${part}_world_size_8_rank_${rank}.pt" ]]
    done
done
if [[ -e "${CONT}/resume_started_at.txt" ]]; then
    echo "Continuation already started: ${CONT}" >&2
    exit 1
fi
idle_count="$(npu-smi info | grep -c 'No running processes found in NPU' || true)"
if [[ "${idle_count}" -ne 8 ]]; then
    echo "Expected eight idle NPUs, found ${idle_count}; refusing to resume" >&2
    exit 1
fi

mkdir -p "${CONT}/generations"
printf '%s\n' "$(date -Is)" > "${CONT}/resume_started_at.txt"
trap 'status=$?; printf "exit_code=%s\nfinished_at=%s\n" "$status" "$(date -Is)" > "${CONT}/exit_status.txt"' EXIT

export MODEL_PATH=/home/chentaoxu/data/models/openPangu-VL-7B
export TRAIN_FILE="${DATA}/train_single_image_max3072_seed42_4000.parquet"
export TEST_FILE="${DATA}/test_single_image_fixed1698.parquet"
export TRAIN_MAX_SAMPLES=-1 VAL_MAX_SAMPLES=-1 FILTER_OVERLONG_PROMPTS=False
export TOTAL_TRAINING_STEPS=930 TRAIN_BATCH_SIZE=64 PPO_MINI_BATCH_SIZE=64 ROLLOUT_N=8
export MAX_PROMPT_LENGTH=7168 MAX_RESPONSE_LENGTH=1024 MAX_MODEL_LENGTH=16384
export PPO_MAX_TOKEN_LEN_PER_GPU=4096 ROLLOUT_GPU_MEM_UTIL=0.60 AGENT_NUM_WORKERS=8
export TEST_FREQ=25 SAVE_FREQ=100 LOG_VAL_GENERATIONS=16 RESUME_MODE=resume_path
export TRAINER_LOGGER='["console","wandb"]' WANDB_MODE=online
export WANDB_RUN_ID=5yhv8vze WANDB_RESUME=must
export PROJECT_NAME=openpangu_vl_virl4k_grpo EXPERIMENT_NAME="${EXPERIMENT}"
export LOG_DIR="${CONT}" CHECKPOINT_DIR="${RUN}/checkpoints"
export VALIDATION_DATA_DIR="${CONT}/generations"
export RAY_TMPDIR=/tmp/r_hif4_sc930
export HCCL_HOST_SOCKET_PORT_RANGE=62000-62050 HCCL_NPU_SOCKET_PORT_RANGE=63000-63050
export PYTHONPATH="/vllm:${PYTHONPATH:-}"

{
    printf 'resumed_at=%s\n' "$(date -Is)"
    printf 'source_checkpoint=%s\ntarget_steps=930\nepochs=15\nsteps_per_epoch=62\n' "${CKPT}"
    printf 'wandb_run_id=%s\ntest_freq=25\nsave_freq=100\n' "${WANDB_RUN_ID}"
    printf 'checkpoint_retention=all\n'
    printf 'train_sha256=%s\n' "$(sha256sum "${TRAIN_FILE}" | awk '{print $1}')"
    printf 'test_sha256=%s\n' "$(sha256sum "${TEST_FILE}" | awk '{print $1}')"
} > "${CONT}/run_manifest.txt"

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
    trainer.resume_from_path="${CKPT}" \
    trainer.total_epochs=15 \
    trainer.max_actor_ckpt_to_keep=null
