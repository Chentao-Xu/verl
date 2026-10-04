#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
WORK=/home/chentaoxu/data/openpangu-eval-work/diagnostics/zero-adv-hif8-hif4-tis372-20261004
SOURCE=/home/chentaoxu/data/openpangu-eval-work/training/virl4k-hif8-hif4-tis-20261001/openpangu_vl_7b_virl4k_hif8_hif4_w4a8_tis_b64n8_e6_seed42_accel_dynbsz4096_conc32
ANALYZER=/home/chentaoxu/openpangu/scripts/eval/diagnose_grpo_zero_advantage.py
[[ ! -e "${WORK}/started.txt" ]] || exit 2
mkdir -p "${WORK}"
date -Is > "${WORK}/started.txt"
exec > >(tee -a "${WORK}/launcher.log") 2>&1
trap 'rc=$?; printf "exit_code=%s\nfinished_at=%s\n" "$rc" "$(date -Is)" > "${WORK}/exit_status.txt"' EXIT
export TRAIN_FILE=/home/chentaoxu/data/datasets/virl39k/processed/subsets/virl4k_seed42_v1/train_single_image_max3072_seed42_4000.parquet
export PYTHONPATH=/vllm:${REPO}:${PYTHONPATH:-}
python "${ANALYZER}" prepare --work "${WORK}" --train "${TRAIN_FILE}"
export TEST_FILE=${WORK}/train_sample256.parquet
export TRAIN_MAX_SAMPLES=-1 VAL_MAX_SAMPLES=-1 FILTER_OVERLONG_PROMPTS=False
export TOTAL_TRAINING_STEPS=1 TRAIN_BATCH_SIZE=64 PPO_MINI_BATCH_SIZE=64 ROLLOUT_N=8
export MAX_PROMPT_LENGTH=7168 MAX_RESPONSE_LENGTH=1024 MAX_MODEL_LENGTH=16384
export PPO_MAX_TOKEN_LEN_PER_GPU=4096 ROLLOUT_GPU_MEM_UTIL=0.60 AGENT_NUM_WORKERS=8
export TEST_FREQ=-1 SAVE_FREQ=-1 RESUME_MODE=resume_path ROLLOUT_ONLY=1
export TRAINER_LOGGER='["console"]' WANDB_MODE=disabled
export EXPERIMENT_NAME=zero_adv_hif8_hif4_tis372 LOG_DIR=${WORK}
export CHECKPOINT_DIR=${WORK}/unused_checkpoints VALIDATION_DATA_DIR=${WORK}/generations
export RAY_TMPDIR=/tmp/r_zero_adv372
export VERL_MXFP8_ACTOR_QAT=0 VERL_MXFP8_ACTOR_TRAIN_SIM=0
export VERL_HIF8_ACTOR_TRAIN_SIM=1 VERL_HIF8_ACTOR_TRAIN_LAYER_RANGE=0:33
export ROLLOUT_QUANTIZATION=hif4_w4a8_sim VERL_MXFP8_SIM_MODE=w8a8
export VERL_LOW_PRECISION_WEIGHT_FORMAT=hif4 VERL_LOW_PRECISION_ACTIVATION_FORMAT=hif8
export VERL_MXFP8_SIM_SCOPE=language_only VERL_MXFP8_SIM_LANGUAGE_LAYER_RANGE=0:33
export VERL_MXFP8_SIM_ACTIVATION_HOOK=apply VERL_MXFP8_SIM_ACTIVATION_GROUPS=all
export HCCL_HOST_SOCKET_PORT_RANGE=62000-62050 HCCL_NPU_SOCKET_PORT_RANGE=63000-63050
bash "${SCRIPT_DIR}/run_openpangu_vl_7b_npu_mxfp8_sim.sh" \
    trainer.resume_from_path="${SOURCE}/checkpoints/global_step_372" \
    trainer.val_only=True actor_rollout_ref.actor.checkpoint.load_contents='[model,extra]' \
    data.val_batch_size=64 data.shuffle=False \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.max_num_seqs=32 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.score_centering_topk=0
python "${ANALYZER}" analyze --work "${WORK}" --generations "${WORK}/generations/372.jsonl"
