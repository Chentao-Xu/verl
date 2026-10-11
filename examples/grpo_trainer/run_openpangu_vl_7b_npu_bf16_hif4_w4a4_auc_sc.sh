#!/usr/bin/env bash
# AUC + PPO mismatch-residual SC. C5/top128, no PG scaling or auxiliary mismatch loss.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
DATA=/home/chentaoxu/data/datasets/virl39k/processed/subsets/virl4k_seed42_v1
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-openpangu_vl_7b_virl4k_bf16_hif4_w4a4_auc_sc128_b64n8_e15_seed42_accel_dynbsz4096_conc32}
export LOG_DIR=${LOG_DIR:-/home/chentaoxu/data/openpangu-eval-work/training/virl4k-bf16-hif4-w4a4-auc-20261011/${EXPERIMENT_NAME}}
if [[ -e "${LOG_DIR}/run_manifest.txt" || -e "${LOG_DIR}/checkpoints" ]]; then
    echo "Refusing to overwrite ${LOG_DIR}" >&2; exit 2
fi
mkdir -p "${LOG_DIR}"
trap 'status=$?; printf "exit_code=%s\nfinished_at=%s\n" "$status" "$(date -Is)" > "${LOG_DIR}/exit_status.txt"' EXIT
export MODEL_PATH=/home/chentaoxu/data/models/openPangu-VL-7B
export TRAIN_FILE=${DATA}/train_single_image_max3072_seed42_4000.parquet
export TEST_FILE=${DATA}/test_single_image_fixed1698.parquet
export TRAIN_MAX_SAMPLES=-1 VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:--1} FILTER_OVERLONG_PROMPTS=False
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-930}
export TRAIN_BATCH_SIZE=64 PPO_MINI_BATCH_SIZE=64 ROLLOUT_N=8
export MAX_PROMPT_LENGTH=7168 MAX_RESPONSE_LENGTH=1024 MAX_MODEL_LENGTH=16384
export PPO_MAX_TOKEN_LEN_PER_GPU=4096 ROLLOUT_GPU_MEM_UTIL=0.60 AGENT_NUM_WORKERS=8
export TEST_FREQ=${TEST_FREQ:-25} SAVE_FREQ=${SAVE_FREQ:-100} LOG_VAL_GENERATIONS=16 RESUME_MODE=disable
export TRAINER_LOGGER=${TRAINER_LOGGER:-'["console","wandb","swanlab"]'} WANDB_MODE=${WANDB_MODE:-online}
export SWANLAB_MODE=online SWANLAB_LOG_DIR=${LOG_DIR}/swanlab
export SWANLAB_RUN_ID=${SWANLAB_RUN_ID:-b16h4a4aucsc15_20261011} SWANLAB_RESUME=allow
export PROJECT_NAME=openpangu_vl_virl4k_grpo
export WANDB_RUN_ID=${WANDB_RUN_ID:-b16h4a4aucsc15_20261011} WANDB_RESUME=never
export CHECKPOINT_DIR=${LOG_DIR}/checkpoints VALIDATION_DATA_DIR=${LOG_DIR}/generations
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/r_aucsc930}
export HCCL_HOST_SOCKET_PORT_RANGE=62100-62150 HCCL_NPU_SOCKET_PORT_RANGE=63100-63150
export PYTHONPATH="/vllm:${PYTHONPATH:-}"
export ROLLOUT_QUANTIZATION=hif4_w4a4_sim
export VERL_LOW_PRECISION_WEIGHT_FORMAT=hif4 VERL_LOW_PRECISION_ACTIVATION_FORMAT=hif4
# Legacy 'w8a8' mode means enable both operand QDQ hooks; actual formats are above.
export VERL_MXFP8_SIM_MODE=w8a8 VERL_MXFP8_SIM_SCOPE=language_only
export VERL_MXFP8_SIM_LANGUAGE_LAYER_RANGE=0:33
export VERL_MXFP8_SIM_ACTIVATION_HOOK=apply VERL_MXFP8_SIM_ACTIVATION_GROUPS=all
export VERL_MXFP8_ACTOR_QAT=0 VERL_MXFP8_ACTOR_TRAIN_SIM=0 VERL_HIF8_ACTOR_TRAIN_SIM=0
{
    printf 'started_at=%s\ngit_head=%s\n' "$(date -Is)" "$(git -c safe.directory="${REPO}" -C "${REPO}" rev-parse HEAD)"
    printf 'actor=bf16\nreference=bf16\nrollout=hif4_w4a4_sim\nweight_qtype=hifx4\nactivation_qtype=hifx4\ngemm=bf16\n'
    printf 'scope=language_layers_0_33\nvision=bf16\nscore_centering_topk=128\nauc=true\nauc_mismatch_loss=false\nauc_pg_scaling=false\nordinary_tis=false\nkl_loss_coef=0.01\nloss_agg_mode=token-mean\n'
    printf 'total_steps=%s\ntrain_batch_size=64\nppo_mini_batch_size=64\nrollout_n=8\nsteps_per_epoch=62\ntotal_epochs=15\n' "${TOTAL_TRAINING_STEPS}"
    printf 'test_freq=%s\nsave_freq=%s\nval_max_samples=%s\n' "${TEST_FREQ}" "${SAVE_FREQ}" "${VAL_MAX_SAMPLES}"
    sha256sum "${TRAIN_FILE}" "${TEST_FILE}"
} > "${LOG_DIR}/run_manifest.txt"
bash "${SCRIPT_DIR}/run_openpangu_vl_7b_npu_mxfp8_sim.sh" \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.fsdp_config.qat.enable=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
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
    actor_rollout_ref.rollout.logprobs_mode=raw_logprobs \
    actor_rollout_ref.actor.policy_loss.loss_mode=auc_sc \
    actor_rollout_ref.actor.policy_loss.auc_mismatch_loss=False \
    algorithm.rollout_correction.rollout_is=null \
    algorithm.rollout_correction.rollout_rs=null \
    trainer.total_epochs=15 trainer.max_actor_ckpt_to_keep=20 "$@"
