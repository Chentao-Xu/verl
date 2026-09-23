#!/usr/bin/env bash
# One-step BF16 Actor + MXFP4 W4A8 rollout score-centering acceptance test.
# Intentionally no WandB, validation, or checkpoint; not a convergence run.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA=/home/chentaoxu/data/datasets/virl39k/processed/subsets/virl4k_seed42_v1

export TRAIN_FILE="${DATA}/train_single_image_max3072_seed42_4000.parquet"
export TEST_FILE="${DATA}/test_single_image_fixed1698.parquet"
export TRAIN_MAX_SAMPLES=16
export VAL_MAX_SAMPLES=8
export TOTAL_TRAINING_STEPS=1
export TRAIN_BATCH_SIZE=8
export PPO_MINI_BATCH_SIZE=8
export ROLLOUT_N=2
export MAX_PROMPT_LENGTH=7168
export MAX_RESPONSE_LENGTH=256
export MAX_MODEL_LENGTH=16384
export PPO_MAX_TOKEN_LEN_PER_GPU=4096
export TEST_FREQ=-1
export SAVE_FREQ=-1
export TRAINER_LOGGER='["console"]'
export WANDB_MODE=disabled
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-openpangu_vl_7b_bf16_mxfp4_score_centering_1step_smoke_20260923}
export LOG_DIR=${LOG_DIR:-/home/chentaoxu/data/openpangu-eval-work/training/score-centering-smoke-20260923}
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/r_sc_smoke}
export PYTHONPATH="/vllm:${PYTHONPATH:-}"

exec "${SCRIPT_DIR}/run_openpangu_vl_7b_npu_bf16_mxfp4_w4a8_smoke.sh" \
    actor_rollout_ref.rollout.score_centering_topk=128 \
    actor_rollout_ref.actor.policy_loss.loss_mode=score_centering \
    algorithm.rollout_correction.rollout_is=null \
    algorithm.rollout_correction.rollout_rs=null \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4096 \
    actor_rollout_ref.rollout.max_num_seqs=32 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.60 \
    trainer.val_before_train=False \
    "$@"
