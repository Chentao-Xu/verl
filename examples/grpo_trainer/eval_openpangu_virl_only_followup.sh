#!/usr/bin/env bash
# ViRL-only follow-up: same saved weights, fixed1698 and BF16 protocol.
set -euo pipefail
REPO=/home/chentaoxu/workspace/verl-mxfp8-formal-200-20260813
RUNNER=${REPO}/examples/grpo_trainer/run_openpangu_vl_7b_npu.sh
MODEL=/home/chentaoxu/data/models/openPangu-VL-7B
DATA=/home/chentaoxu/data/datasets/virl39k/processed/subsets/virl4k_seed42_v1
RUN=${EVAL_RUN:?}
WORK=${EVAL_WORK:?}
EVAL_LABEL=${EVAL_LABEL:?}
VIRL_RESULTS=${WORK}/virl_results.tsv
STATUS=${WORK}/status.log
mkdir -p "${WORK}"
exec > >(tee -a "${WORK}/virl_only_20261007.log") 2>&1
unset VERL_MXFP8_SIM_MODE VERL_MXFP8_SIM_SCOPE VERL_MXFP8_SIM_LANGUAGE_LAYER_RANGE
unset VERL_MXFP8_ACTOR_QAT VERL_MXFP8_ACTOR_SIM VERL_MXFP8_SIM VERL_HIF8_ACTOR_TRAIN_SIM
unset VERL_LOW_PRECISION_SIM_FORMAT VERL_LOW_PRECISION_WEIGHT_FORMAT VERL_LOW_PRECISION_ACTIVATION_FORMAT
export PYTHONPATH=/vllm:${REPO}:${PYTHONPATH:-}
export EVAL_LOAD_CONTENTS='[model,extra]'
idle_npus() {
    npu-smi info | grep -c 'No running processes found in NPU' || true
}

wait_idle() {
    local attempt
    for attempt in {1..60}; do
        [[ "$(idle_npus)" -eq 8 ]] && return 0
        sleep 5
    done
    return 1
}

cleanup_ray() {
    wait_idle
}

run_virl() {
    local step=$1
    local ckpt=$2
    local out=${WORK}/step${step}/virl1698_bf16
    local label=${EVAL_LABEL}_step${step}
    if awk -F '\t' -v key="${step}" 'NR>1 && $1==key {found=1} END{exit !found}' "${VIRL_RESULTS}" 2>/dev/null; then
        echo "$(date -Is) SKIP ViRL step${step}"
        return 0
    fi
    [[ "$(idle_npus)" -eq 8 ]]
    mkdir -p "${out}"
    local started elapsed log generations samples acc reward
    started=$(date +%s)
    cd "${REPO}"
    env \
        MODEL_PATH="${MODEL}" TRAIN_FILE="${DATA}/train_single_image_max3072_seed42_4000.parquet" \
        TEST_FILE="${DATA}/test_single_image_fixed1698.parquet" NDEVICES_PER_NODE=8 \
        ROLLOUT_TP=1 ROLLOUT_N=4 ROLLOUT_GPU_MEM_UTIL=0.45 TRAIN_BATCH_SIZE=16 \
        PPO_MINI_BATCH_SIZE=16 TRAIN_MAX_SAMPLES=-1 VAL_MAX_SAMPLES=-1 ROLLOUT_ONLY=1 \
        FILTER_OVERLONG_PROMPTS=False MAX_PROMPT_LENGTH=7168 MAX_RESPONSE_LENGTH=1024 \
        MAX_MODEL_LENGTH=16384 PPO_MAX_TOKEN_LEN_PER_GPU=4096 ACTOR_ATTN_IMPLEMENTATION=sdpa \
        VERL_FORCE_WEIGHT_SYNC_SHM=1 VERL_ROLLOUT_LOGPROB_CHECK=1 VERL_WEIGHT_SYNC_AUDIT=1 \
        VERL_MXFP8_ACTOR_TRAIN_SIM=0 VERL_HIF8_ACTOR_TRAIN_SIM=0 \
        TRAINER_LOGGER='["console"]' WANDB_MODE=disabled PROJECT_NAME=openpangu_eval \
        EXPERIMENT_NAME="${label}" LOG_DIR="${out}" CHECKPOINT_DIR="${out}/checkpoint_stub" \
        VALIDATION_DATA_DIR="${out}/generations" RAY_TMPDIR="/tmp/r_virl4k_eval_${step}" \
        HCCL_HOST_SOCKET_PORT_RANGE=58000-58050 HCCL_NPU_SOCKET_PORT_RANGE=59000-59050 \
        LOG_VAL_GENERATIONS=8 TEST_FREQ=-1 SAVE_FREQ=-1 RESUME_MODE=resume_path \
        bash "${RUNNER}" hydra.run.dir="${out}/hydra" trainer.resume_from_path="${ckpt}" trainer.resume_mode=resume_path \
        actor_rollout_ref.rollout.quantization=null \
        "actor_rollout_ref.actor.checkpoint.load_contents=${EVAL_LOAD_CONTENTS:-[model,optimizer,extra]}"
    log=${out}/${label}.log
    generations=$(find "${out}/generations" -maxdepth 1 -type f -name '*.jsonl' | sort -V | tail -n 1)
    samples=$(wc -l < "${generations}")
    acc=$(grep -oE 'val-core/[^ ]+/acc/mean@1:np\.float64\([^)]+' "${log}" | tail -n1 | sed 's/.*(//')
    reward=$(grep -oE 'val-aux/[^ ]+/reward/mean@1:np\.float64\([^)]+' "${log}" | tail -n1 | sed 's/.*(//')
    elapsed=$(( $(date +%s) - started ))
    [[ "${samples}" -eq 1698 && -n "${acc}" ]]
    printf '%s\t%s\t%s\t%s\t%s\n' "${step}" "${samples}" "${acc}" "${reward}" "${elapsed}" >> "${VIRL_RESULTS}"
    cleanup_ray
}

[[ "$(idle_npus)" -eq 8 ]] || { echo "Expected eight idle NPUs"; exit 3; }
[[ -f "${VIRL_RESULTS}" ]] || printf 'step\tsamples\tacc_mean_at_1\treward_mean_at_1\telapsed_seconds\n' > "${VIRL_RESULTS}"
read -r -a STEPS <<< "${EVAL_STEPS:?}"
for step in "${STEPS[@]}"; do
    ckpt=${RUN}/checkpoints/global_step_${step}
    test -f "${ckpt}/.sync_complete"
    test -f "${ckpt}/data.pt"
    echo "$(date -Is) VIRL_ONLY_START step${step}" | tee -a "${STATUS}"
    run_virl "${step}" "${ckpt}"
    echo "$(date -Is) VIRL_ONLY_DONE step${step}" | tee -a "${STATUS}"
done
