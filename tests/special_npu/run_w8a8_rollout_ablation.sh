#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "usage: $0 <activation-hook> <activation-groups> [visible-device] [mode]" >&2
  exit 2
fi

hook=$1
groups=$2
device=${3:-0}
mode=${4:-w8a8}
tag="${mode}_${hook}_${groups//,/_}"
repo=/home/chentaoxu/workspace/verl-mxfp8-sim-20260812
out=/home/chentaoxu/data/openpangu-vl-npu/mxfp8-sim-20260812/w8a8_hook_ablation_20260813

export ASCEND_RT_VISIBLE_DEVICES="$device"
export PYTHONPATH="/home/chentaoxu/quant_dequant:${repo}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/home/chentaoxu/quant_dequant/quant_cy_npu/cusrc:/usr/local/python3.11.15/lib/python3.11/site-packages/torch_npu/lib:/usr/local/python3.11.15/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "$out"
cd "$repo"
python tests/special_npu/test_openpangu_logprob_parity.py \
  --phase rollout \
  --model-path /home/chentaoxu/data/models/openPangu-VL-7B \
  --dataset-path /home/chentaoxu/data/datasets/virl39k/processed/val_single_image_seed42_raw256_max3072.parquet \
  --fixture-path "$out/fixture_${tag}.json" \
  --sample-index 0 \
  --max-model-len 2048 \
  --max-tokens 32 \
  --rollout-mxfp8 "$mode" \
  --rollout-mxfp8-scope language_only \
  --rollout-mxfp8-layer-range 0:33 \
  --rollout-mxfp8-activation-hook "$hook" \
  --rollout-mxfp8-activation-groups "$groups"
