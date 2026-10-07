#!/usr/bin/env bash
# Fixed training-subset snapshot; inference only, no optimizer or checkpoint writes.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
WORK=${DIAG_WORK:?DIAG_WORK required}
export MODEL_PATH=${DIAG_MODEL:?DIAG_MODEL required}
[[ ! -e "${WORK}/started.txt" ]] || exit 2
test -s "${MODEL_PATH}/config.json"
mkdir -p "${WORK}"
date -Is > "${WORK}/started.txt"
exec > >(tee -a "${WORK}/launcher.log") 2>&1
trap 'rc=$?; printf "exit_code=%s\nfinished_at=%s\n" "$rc" "$(date -Is)" > "${WORK}/exit_status.txt"' EXIT
ANALYZER=/home/chentaoxu/openpangu/scripts/eval/diagnose_grpo_zero_advantage.py
export TRAIN_FILE=/home/chentaoxu/data/datasets/virl39k/processed/subsets/virl4k_seed42_v1/train_single_image_max3072_seed42_4000.parquet
export PYTHONPATH=/vllm:${REPO}:${PYTHONPATH:-}
python "${ANALYZER}" prepare --work "${WORK}" --train "${TRAIN_FILE}"
cmp "${WORK}/sample_indices.json" /home/chentaoxu/data/openpangu-eval-work/diagnostics/zero-adv-hif8-hif4-tis372-20261004/sample_indices.json
printf 'model_path=%s\nprecision=bf16\nquantization=null\nseed=42\nn=8\ntemperature=1\nresponse_length=1024\n' "${MODEL_PATH}" > "${WORK}/manifest.txt"
sha256sum "${MODEL_PATH}/config.json" "${TRAIN_FILE}" >> "${WORK}/manifest.txt"
export TEST_FILE=${WORK}/train_sample256.parquet
export TRAIN_MAX_SAMPLES=-1 VAL_MAX_SAMPLES=-1 FILTER_OVERLONG_PROMPTS=False
export TOTAL_TRAINING_STEPS=1 TRAIN_BATCH_SIZE=64 PPO_MINI_BATCH_SIZE=64 ROLLOUT_N=8
export MAX_PROMPT_LENGTH=7168 MAX_RESPONSE_LENGTH=1024 MAX_MODEL_LENGTH=16384
export PPO_MAX_TOKEN_LEN_PER_GPU=4096 ROLLOUT_GPU_MEM_UTIL=0.60 AGENT_NUM_WORKERS=8
export TEST_FREQ=-1 SAVE_FREQ=-1 RESUME_MODE=disable ROLLOUT_ONLY=1
export TRAINER_LOGGER='["console"]' WANDB_MODE=disabled
export EXPERIMENT_NAME=${DIAG_LABEL:?DIAG_LABEL required} LOG_DIR=${WORK}
export CHECKPOINT_DIR=${WORK}/unused_checkpoints VALIDATION_DATA_DIR=${WORK}/generations
export RAY_TMPDIR=/tmp/r_bf16_diag_${DIAG_LABEL}
export VERL_MXFP8_ACTOR_QAT=0 VERL_MXFP8_ACTOR_TRAIN_SIM=0 VERL_HIF8_ACTOR_TRAIN_SIM=0
export ROLLOUT_QUANTIZATION=null
export HCCL_HOST_SOCKET_PORT_RANGE=62000-62050 HCCL_NPU_SOCKET_PORT_RANGE=63000-63050
bash "${SCRIPT_DIR}/run_openpangu_vl_7b_npu_mxfp8_sim.sh" \
    trainer.resume_from_path=null trainer.val_only=True \
    actor_rollout_ref.actor.fsdp_config.qat.enable=False \
    actor_rollout_ref.rollout.quantization=null \
    data.val_batch_size=64 data.shuffle=False data.seed=42 \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.max_num_seqs=32 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.score_centering_topk=0
python "${ANALYZER}" analyze --work "${WORK}" --generations "${WORK}/generations/0.jsonl"
