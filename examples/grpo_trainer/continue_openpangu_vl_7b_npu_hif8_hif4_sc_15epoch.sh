#!/usr/bin/env bash
# Resume the existing HiF8–HiF4 SC run after step372 to step930.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
DATA=/home/chentaoxu/data/datasets/virl39k/processed/subsets/virl4k_seed42_v1
EXPERIMENT=openpangu_vl_7b_virl4k_hif8_hif4_w4a8_sc128_b64n8_e6_seed42_accel_dynbsz4096_conc32
RUN=/home/chentaoxu/data/openpangu-eval-work/training/virl4k-hif8-hif4-sc-20261002/${EXPERIMENT}
SOURCE_RUN=${RUN}
CKPT=${SOURCE_RUN}/checkpoints/global_step_372
# Wait without interrupting the existing six-epoch run.
for ((attempt=0; attempt<2880; attempt++)); do
    [[ -f "${SOURCE_RUN}/exit_status.txt" ]] && break
    sleep 60
done
grep -qx 'exit_code=0' "${SOURCE_RUN}/exit_status.txt"
[[ -s "${CKPT}/data.pt" ]]
for rank in {0..7}; do
    for part in model optim extra_state; do
        [[ -s "${CKPT}/actor/${part}_world_size_8_rank_${rank}.pt" ]]
    done
done
for ((attempt=0; attempt<60; attempt++)); do
    [[ "$(npu-smi info | grep -c 'No running processes found in NPU' || true)" == 8 ]] && break
    sleep 5
done
[[ "$(npu-smi info | grep -c 'No running processes found in NPU' || true)" == 8 ]]
RUN=${SOURCE_RUN}/continuation_step372_to930
[[ ! -e "${RUN}/run_manifest.txt" ]] || { echo "Refusing existing run: ${RUN}"; exit 2; }
mkdir -p "${RUN}/generations"
exec > >(tee -a "${RUN}/launcher.log") 2>&1
trap 'rc=$?; printf "exit_code=%s\nfinished_at=%s\n" "$rc" "$(date -Is)" > "${RUN}/exit_status.txt"' EXIT

export MODEL_PATH=/home/chentaoxu/data/models/openPangu-VL-7B
export TRAIN_FILE=${DATA}/train_single_image_max3072_seed42_4000.parquet
export TEST_FILE=${DATA}/test_single_image_fixed1698.parquet
export TRAIN_MAX_SAMPLES=-1 VAL_MAX_SAMPLES=-1 FILTER_OVERLONG_PROMPTS=False
export TOTAL_TRAINING_STEPS=930 TRAIN_BATCH_SIZE=64 PPO_MINI_BATCH_SIZE=64 ROLLOUT_N=8
export NDEVICES_PER_NODE=8 ROLLOUT_TP=1
export MAX_PROMPT_LENGTH=7168 MAX_RESPONSE_LENGTH=1024 MAX_MODEL_LENGTH=16384
export PPO_MAX_TOKEN_LEN_PER_GPU=4096 ROLLOUT_GPU_MEM_UTIL=0.60 AGENT_NUM_WORKERS=8
export TEST_FREQ=25 SAVE_FREQ=100 LOG_VAL_GENERATIONS=16 RESUME_MODE=resume_path
export TRAINER_LOGGER='["console","wandb"]' WANDB_MODE=online
export PROJECT_NAME=openpangu_vl_virl4k_grpo EXPERIMENT_NAME=${EXPERIMENT}
export LOG_DIR=${RUN} CHECKPOINT_DIR=${SOURCE_RUN}/checkpoints VALIDATION_DATA_DIR=${RUN}/generations
export RAY_TMPDIR=/tmp/r_h8h4_sc930
export HCCL_HOST_SOCKET_PORT_RANGE=62000-62050 HCCL_NPU_SOCKET_PORT_RANGE=63000-63050
export PYTHONPATH=/vllm:${PYTHONPATH:-}
export ROLLOUT_QUANTIZATION=hif4_w4a8_sim
export VERL_LOW_PRECISION_WEIGHT_FORMAT=hif4 VERL_LOW_PRECISION_ACTIVATION_FORMAT=hif8
export VERL_MXFP8_SIM_MODE=w8a8 VERL_MXFP8_SIM_SCOPE=language_only
export VERL_MXFP8_SIM_LANGUAGE_LAYER_RANGE=0:33 VERL_MXFP8_SIM_ACTIVATION_HOOK=apply VERL_MXFP8_SIM_ACTIVATION_GROUPS=all
export VERL_MXFP8_ACTOR_QAT=0 VERL_MXFP8_ACTOR_TRAIN_SIM=0
export VERL_HIF8_ACTOR_TRAIN_SIM=1 VERL_HIF8_ACTOR_TRAIN_LAYER_RANGE=0:33
export WANDB_RUN_ID=h8h4sc6_20261002 WANDB_RESUME=must
{
    printf 'started_at=%s\nhostname=%s\n' "$(date -Is)" "$(hostname)"
    printf 'git_head=%s\n' "$(git -c safe.directory="${REPO}" -C "${REPO}" rev-parse HEAD)"
    printf 'actor=non_qat_hif8_forward_backward\nrollout=hif4_w4a8\nreference=bf16\nvision=bf16\n'
    printf 'train_rows=4000\nsteps_per_epoch=62\nepochs=15\nsteps=930\nbatch=64\nppo_mini_batch=64\nrollout_n=8\n'
    printf 'score_centering=true\nscore_centering_topk=128\nrollout_is=null\nrollout_is_threshold=2.0\nrollout_is_batch_normalize=false\nrollout_rs=null\n'
    printf 'lr=1e-6\nkl_coef=0.01\nseed=42\nshuffle=true\nprompt=7168\nresponse=1024\n'
    printf 'test_freq=25\nval_samples=1698\nsave_freq=100\nkeep_all_checkpoints=true\n'
    printf 'actor_dynamic_token_budget=4096\nrollout_max_num_seqs=32\nrollout_max_num_batched_tokens=32768\n'
    sha256sum "${TRAIN_FILE}" "${TEST_FILE}"
} > "${RUN}/run_manifest.txt"
bash "${SCRIPT_DIR}/run_openpangu_vl_7b_npu_mxfp8_sim.sh" \
    actor_rollout_ref.actor.fsdp_config.qat.enable=False \
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
    actor_rollout_ref.rollout.score_centering_topk=128 \
    actor_rollout_ref.actor.policy_loss.loss_mode=score_centering \
    algorithm.rollout_correction.rollout_is=null \
    algorithm.rollout_correction.rollout_rs=null \
    trainer.resume_from_path="${CKPT}" trainer.total_epochs=15 trainer.max_actor_ckpt_to_keep=null
