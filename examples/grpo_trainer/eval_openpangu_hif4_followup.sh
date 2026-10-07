#!/usr/bin/env bash
set -euo pipefail

REPO=/home/chentaoxu/workspace/verl-mxfp8-formal-200-20260813
RUNNER=${REPO}/examples/grpo_trainer/run_openpangu_vl_7b_npu.sh
MODEL=/home/chentaoxu/data/models/openPangu-VL-7B
DATA=/home/chentaoxu/data/datasets/virl39k/processed/subsets/virl4k_seed42_v1
RUN=${EVAL_RUN:-/home/chentaoxu/data/openpangu-vl-npu/virl4k-precision-accel-20260913/openpangu_vl_7b_virl4k_hif8_hif8_b64n8_e9_seed42_accel_dynbsz4096_conc32}
WORK=${EVAL_WORK:-/home/chentaoxu/data/openpangu-eval-work/virl4k-unified-bf16-20260921/hif8_hif8}
EVAL_LABEL=${EVAL_LABEL:-hif8_hif8}
MERGER=/home/chentaoxu/merge_openpangu_fsdp_to_hf.py
LMMS_RUNNER=/home/chentaoxu/workspace/verl-hif4-w4a8-20260925/examples/grpo_trainer/eval_lmms_bounded_repair.sh
STATUS=${WORK}/status.log
VIRL_RESULTS=${WORK}/virl_results.tsv
BASE_PORT=18100
read -r -a STEPS <<< "${EVAL_STEPS:-200 400 600 800 900 930}"
GPQA_STEPS=${EVAL_GPQA_STEPS:-900 930}

mkdir -p "${WORK}"
if [[ ! -e "${WORK}/run_manifest.txt" ]]; then
    {
        printf 'started_at=%s\n' "$(date -Is)"
        printf 'hostname=%s\n' "$(hostname)"
        printf 'source_checkpoint_run=%s\n' "${RUN}"
        printf 'evaluation_label=%s\n' "${EVAL_LABEL}"
        printf 'steps=%s\n' "${STEPS[*]}"
        printf 'virl_test=%s\n' "${DATA}/test_single_image_fixed1698.parquet"
        printf 'inference_precision=bf16\nrollout_quantization=none\n'
        printf 'downstream=mmmu_pangu_official,mathvista_testmini_rules,ocrbench,geometry3k_freeform\n'
        printf 'gpqa_steps=%s\n' "${GPQA_STEPS}"
    } > "${WORK}/run_manifest.txt"
fi
exec > >(tee -a "${WORK}/pipeline.log") 2>&1

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

stop_servers() {
    local pid
    for pid in "${SERVER_PIDS[@]:-}"; do
        kill -TERM -- "-${pid}" 2>/dev/null || true
    done
    for pid in "${SERVER_PIDS[@]:-}"; do
        wait "${pid}" 2>/dev/null || true
    done
    SERVER_PIDS=()
    wait_idle
}

start_servers() {
    local merged=$1
    local model_name=$2
    local server_dir=$3
    local card port pid attempt
    mkdir -p "${server_dir}"
    SERVER_PIDS=()
    set +u
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    set -u
    export PYTHONPATH=/vllm:${REPO}:${PYTHONPATH:-}
    export OMP_NUM_THREADS=4
    for card in {0..7}; do
        port=$((BASE_PORT + card))
        setsid env ASCEND_RT_VISIBLE_DEVICES=${card} \
            python -m vllm.entrypoints.openai.api_server \
            --model "${merged}" --served-model-name "${model_name}" \
            --trust-remote-code \
            --hf-overrides '{"architectures":["OpenPanguVLForConditionalGeneration"]}' \
            --dtype bfloat16 --tensor-parallel-size 1 --enforce-eager \
            --max-model-len 65536 --max-num-batched-tokens 8192 \
            --max-num-seqs 16 --gpu-memory-utilization 0.8 \
            --limit-mm-per-prompt '{"image":7,"video":0}' \
            --host 127.0.0.1 --port "${port}" \
            > "${server_dir}/card${card}.log" 2>&1 &
        pid=$!
        SERVER_PIDS+=("${pid}")
        sleep 2
    done
    local py=/home/chentaoxu/workspace/lmms-eval/.venv-py311/bin/python
    for attempt in {1..90}; do
        if MODEL_NAME=${model_name} BASE_PORT=${BASE_PORT} "${py}" -c \
            'import json,os,urllib.request; name=os.environ["MODEL_NAME"]; p=int(os.environ["BASE_PORT"]); [next(m for m in json.load(urllib.request.urlopen(f"http://127.0.0.1:{p+i}/v1/models",timeout=5))["data"] if m["id"]==name) for i in range(8)]' \
            >/dev/null 2>&1; then
            echo "$(date -Is) all eight TP1 servers ready"
            return 0
        fi
        sleep 10
    done
    echo "TP1x8 services failed to become ready" >&2
    return 1
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

[[ "$(idle_npus)" -eq 8 ]] || { echo "Expected eight idle NPUs" >&2; exit 3; }
[[ -f "${VIRL_RESULTS}" ]] || printf 'step\tsamples\tacc_mean_at_1\treward_mean_at_1\telapsed_seconds\n' > "${VIRL_RESULTS}"
unset VERL_MXFP8_SIM_MODE VERL_MXFP8_SIM_SCOPE VERL_MXFP8_SIM_LANGUAGE_LAYER_RANGE || true
unset VERL_MXFP8_ACTOR_QAT VERL_MXFP8_ACTOR_SIM VERL_MXFP8_SIM || true
unset VERL_HIF8_ACTOR_TRAIN_SIM VERL_LOW_PRECISION_SIM_FORMAT || true
SERVER_PIDS=()
trap stop_servers EXIT

for step in "${STEPS[@]}"; do
    ckpt=${RUN}/checkpoints/global_step_${step}
    if [[ "${EVAL_WAIT_FOR_SYNC:-0}" == 1 ]]; then
        for ((attempt=0; attempt<720; attempt++)); do
            [[ -f "${ckpt}/.sync_complete" ]] && break
            sleep 5
        done
        [[ -f "${ckpt}/.sync_complete" ]] || { echo "Checkpoint sync timed out: ${ckpt}"; exit 4; }
    fi
    actor=${ckpt}/actor
    merged=${WORK}/step${step}/merged_hf
    model_name=openpangu-virl4k-${EVAL_LABEL}-step${step}-bf16
    echo "$(date -Is) START step${step}" | tee -a "${STATUS}"
    test -f "${ckpt}/data.pt"
    test -f "${actor}/model_world_size_8_rank_0.pt"
    run_virl "${step}" "${ckpt}"

    if [[ ! -s "${merged}/model.safetensors" ]]; then
        mkdir -p "${merged}"
        cd "${REPO}"
        python "${MERGER}" --actor-dir "${actor}" --base-model-dir "${MODEL}" --output-dir "${merged}"
    fi
    start_servers "${merged}" "${model_name}" "${WORK}/step${step}/servers"
    BASE_PORT=${BASE_PORT} bash "${LMMS_RUNNER}" mmmu_pangu_official 900 4096 "${WORK}/step${step}/lmms/mmmu_pangu_official" "${model_name}"
    BASE_PORT=${BASE_PORT} bash "${LMMS_RUNNER}" mathvista_testmini_rules 1000 4096 "${WORK}/step${step}/lmms/mathvista_testmini_rules" "${model_name}"
    BASE_PORT=${BASE_PORT} bash "${LMMS_RUNNER}" ocrbench 1000 128 "${WORK}/step${step}/lmms/ocrbench" "${model_name}"
    BASE_PORT=${BASE_PORT} bash "${LMMS_RUNNER}" geometry3k_freeform 601 2048 "${WORK}/step${step}/lmms/geometry3k_freeform" "${model_name}"
    if [[ " ${GPQA_STEPS} " == *" ${step} "* ]]; then
        BASE_PORT=${BASE_PORT} bash "${LMMS_RUNNER}" gpqa_diamond_cot_zeroshot 198 4096 "${WORK}/step${step}/lmms/gpqa_diamond_cot_zeroshot" "${model_name}"
    fi
    stop_servers
    echo "$(date -Is) DONE step${step}" | tee -a "${STATUS}"
done

echo "$(date -Is) ALL_ATTEMPTED ${EVAL_LABEL}" | tee -a "${STATUS}"
