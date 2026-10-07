#!/usr/bin/env bash
# Same evaluation protocol; one bounded retry wave for failed requests only.
set -euo pipefail
TASK=$1 TOTAL=$2 TOKENS=$3 OUT=$4 MODEL_NAME=$5
BASE_PORT=${BASE_PORT:-18100}
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PY=/home/chentaoxu/workspace/lmms-eval/.venv-py311/bin/python
SUMMARY=${HERE}/summarize_lmms_jsonl_safe.py
mkdir -p "${OUT}"
if [[ -s "${OUT}/summary.json" ]] && "${PY}" -c 'import json,sys; s=json.load(open(sys.argv[1])); sys.exit(0 if s["status"]=="complete" and not s["failed_doc_ids"] else 1)' "${OUT}/summary.json"; then
    echo "SKIP complete ${OUT}"
    exit 0
fi
if [[ ! -s "${OUT}/summary.json" ]]; then
    bash /home/chentaoxu/openpangu/scripts/eval/run_lmms_task_tp1x8.sh "$@" || echo "Initial runner returned nonzero; checking actual sample files"
fi
[[ ! -s "${OUT}/summary.json" ]] || cp -n "${OUT}/summary.json" "${OUT}/summary_before_repair_20261007.json"
"${PY}" "${SUMMARY}" --root "${OUT}" --task "${TASK}" --total "${TOTAL}" --model "${MODEL_NAME}"
ids=$("${PY}" -c 'import json,sys; print(" ".join(map(str,json.load(open(sys.argv[1]))["failed_doc_ids"])))' "${OUT}/summary.json")
[[ -n "$ids" ]] || exit 0
read -r -a IDS <<< "$ids"
# Avoid spending hours retrying systemic failures. Preserve incomplete status.
if (( ${#IDS[@]} > 16 )); then
    echo "NEEDS_REPAIR: too many failures (${#IDS[@]}), skip retry wave"
    exit 0
fi
export HF_DATASETS_CACHE=/home/chentaoxu/workspace/lmms-eval/data/hf_cache HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES='' OPENAI_API_KEY=EMPTY NO_PROXY=127.0.0.1,localhost
cd /home/chentaoxu/workspace/lmms-eval
PIDS=()
for i in "${!IDS[@]}"; do
    id=${IDS[$i]}
    dest=${OUT}/repairs_20261007/doc${id}
    [[ ! -e "$dest/run.log" ]] || continue
    mkdir -p "$dest"
    entry=(-m lmms_eval)
    [[ "$TASK" != mmmu_pangu_official ]] || entry=(/home/chentaoxu/lmms_mmmu_official_entry.py)
    timeout 420 "${PY}" "${entry[@]}" --model openai \
        --model_args "model_version=${MODEL_NAME},base_url=http://127.0.0.1:$((BASE_PORT+i%8))/v1,api_key=EMPTY,num_concurrent=1,timeout=300,max_retries=0,httpx_trust_env=False" \
        --include_path /home/chentaoxu/lmms_custom_tasks --tasks "${TASK}" \
        --offset "$id" --limit 1 --seed 42 \
        --gen_kwargs "temperature=0,max_new_tokens=${TOKENS}" \
        --log_samples --output_path "$dest" > "$dest/run.log" 2>&1 &
    PIDS+=("$!")
    if (( (i+1)%8 == 0 )); then
        for pid in "${PIDS[@]}"; do wait "$pid" || true; done
        PIDS=()
    fi
done
for pid in "${PIDS[@]}"; do wait "$pid" || true; done
"${PY}" "${SUMMARY}" --root "${OUT}" --repairs-root "${OUT}/repairs_20261007" \
    --task "${TASK}" --total "${TOTAL}" --model "${MODEL_NAME}"
# The outer queue validates all summaries; unrepaired requests do not block
# other checkpoints, and are never labelled complete.
