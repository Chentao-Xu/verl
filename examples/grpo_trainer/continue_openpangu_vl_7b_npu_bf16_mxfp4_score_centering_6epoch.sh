#!/usr/bin/env bash
# Wait for the 200-step acceptance run, then resume its FSDP checkpoint to 6 epochs.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="${SCRIPT_DIR}/run_openpangu_vl_7b_npu_bf16_mxfp4_w4a8_smoke.sh"
DATA=/home/chentaoxu/data/datasets/virl39k/processed/subsets/virl4k_seed42_v1
ROOT=/home/chentaoxu/data/openpangu-eval-work/training/virl4k-bf16-mxfp4-score-centering-20260923
EXPERIMENT=openpangu_vl_7b_virl4k_bf16_mxfp4_w4a8_sc128_b64n8_s200_seed42_accel_dynbsz4096_conc32
RUN="${ROOT}/${EXPERIMENT}"
CONT="${RUN}/continuation_step200_to372"
CKPT="${RUN}/checkpoints/global_step_200"
mkdir -p "${CONT}/generations"

for ((attempt = 0; attempt < 600; attempt++)); do
    if [[ -f "${RUN}/exit_status.txt" ]]; then
        break
    fi
    sleep 60
done
if [[ ! -f "${RUN}/exit_status.txt" ]] || ! grep -qx 'exit_code=0' "${RUN}/exit_status.txt"; then
    echo "200-step run did not finish successfully; refusing to resume" >&2
    exit 1
fi
if [[ ! -f "${CKPT}/data.pt" ]] || [[ ! -d "${CKPT}/actor" ]]; then
    echo "Incomplete step-200 checkpoint: ${CKPT}" >&2
    exit 1
fi
if [[ -e "${CONT}/resume_started_at.txt" ]]; then
    echo "Continuation already started; refusing a duplicate launch" >&2
    exit 1
fi

# The original process can take a moment to release its devices after its exit marker.
for ((attempt = 0; attempt < 20; attempt++)); do
    idle_count="$(npu-smi info | grep -c 'No running processes found in NPU' || true)"
    if [[ "${idle_count}" -eq 8 ]]; then
        break
    fi
    sleep 30
done
if [[ "${idle_count}" -ne 8 ]]; then
    echo "NPUs are occupied after the 200-step run; refusing to start continuation" >&2
    exit 1
fi

printf '%s\n' "$(date -Is)" > "${CONT}/resume_started_at.txt"
trap 'status=$?; printf "exit_code=%s\nfinished_at=%s\n" "$status" "$(date -Is)" > "${CONT}/exit_status.txt"' EXIT

export MODEL_PATH=/home/chentaoxu/data/models/openPangu-VL-7B
export TRAIN_FILE="${DATA}/train_single_image_max3072_seed42_4000.parquet"
export TEST_FILE="${DATA}/test_single_image_fixed1698.parquet"
export TRAIN_MAX_SAMPLES=-1 VAL_MAX_SAMPLES=-1 FILTER_OVERLONG_PROMPTS=False
export TOTAL_TRAINING_STEPS=372 TRAIN_BATCH_SIZE=64 PPO_MINI_BATCH_SIZE=64 ROLLOUT_N=8
export MAX_PROMPT_LENGTH=7168 MAX_RESPONSE_LENGTH=1024 MAX_MODEL_LENGTH=16384
export PPO_MAX_TOKEN_LEN_PER_GPU=4096 ROLLOUT_GPU_MEM_UTIL=0.60 AGENT_NUM_WORKERS=8
export TEST_FREQ=25 SAVE_FREQ=100 LOG_VAL_GENERATIONS=16 RESUME_MODE=resume_path
export TRAINER_LOGGER='["console","wandb"]' WANDB_MODE=online
export WANDB_RUN_ID=ac7lhz25 WANDB_RESUME=must
export PROJECT_NAME=openpangu_vl_virl4k_grpo EXPERIMENT_NAME="${EXPERIMENT}"
export LOG_DIR="${CONT}" CHECKPOINT_DIR="${RUN}/checkpoints"
export VALIDATION_DATA_DIR="${CONT}/generations"
export RAY_TMPDIR=/tmp/r_sc372
export HCCL_HOST_SOCKET_PORT_RANGE=62000-62050 HCCL_NPU_SOCKET_PORT_RANGE=63000-63050
export PYTHONPATH="/vllm:${PYTHONPATH:-}"

{
    printf 'resumed_at=%s\n' "$(date -Is)"
    printf 'source_checkpoint=%s\n' "${CKPT}"
    printf 'target_steps=372\nepochs=6\nsteps_per_epoch=62\n'
    printf 'wandb_run_id=%s\n' "${WANDB_RUN_ID}"
    printf 'test_freq=25\nsave_freq=100\n'
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
    trainer.total_epochs=6 \
    trainer.max_actor_ckpt_to_keep=10
