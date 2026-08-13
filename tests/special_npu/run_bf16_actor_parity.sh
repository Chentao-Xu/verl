#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <fixture-path> <result-path> <visible-device>" >&2
  exit 2
fi

repo=/home/chentaoxu/workspace/verl-mxfp8-sim-20260812
export ASCEND_RT_VISIBLE_DEVICES="$3"
export PYTHONPATH="${repo}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/usr/local/python3.11.15/lib/python3.11/site-packages/torch_npu/lib:/usr/local/python3.11.15/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
cd "$repo"
python tests/special_npu/test_openpangu_logprob_parity.py \
  --phase actor \
  --model-path /home/chentaoxu/data/models/openPangu-VL-7B \
  --dataset-path /home/chentaoxu/data/datasets/virl39k/processed/val_single_image_seed42_raw256_max3072.parquet \
  --fixture-path "$1" \
  --result-path "$2" \
  --actor-attention sdpa \
  --actor-position-ids processor \
  --actor-mxfp8 none \
  --mean-threshold 99 \
  --p95-threshold 99 \
  --max-threshold 99 \
  --pearson-threshold -1
