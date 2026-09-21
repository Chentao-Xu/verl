#!/usr/bin/env bash
# BF16 Actor + MXFP4-weight/MXFP8-activation rollout numerical smoke test.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export ROLLOUT_QUANTIZATION=mxfp4_w4a8_sim
export VERL_LOW_PRECISION_WEIGHT_FORMAT=mxfp4
export VERL_LOW_PRECISION_ACTIVATION_FORMAT=mxfp8
export VERL_MXFP8_SIM_MODE=w8a8
export VERL_MXFP8_SIM_SCOPE=language_only
export VERL_MXFP8_SIM_LANGUAGE_LAYER_RANGE=0:33
export VERL_MXFP8_SIM_ACTIVATION_HOOK=apply
export VERL_MXFP8_SIM_ACTIVATION_GROUPS=all

# Keep the Actor entirely BF16: no non-QAT simulator and no QAT.
export VERL_MXFP8_ACTOR_QAT=0
export VERL_MXFP8_ACTOR_TRAIN_SIM=0
export VERL_HIF8_ACTOR_TRAIN_SIM=0

export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-1}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
export ROLLOUT_N=${ROLLOUT_N:-2}
export TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-16}
export VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-8}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-64}
export TEST_FREQ=${TEST_FREQ:--1}
export SAVE_FREQ=${SAVE_FREQ:--1}
export TRAINER_LOGGER=${TRAINER_LOGGER:-'["console"]'}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-openpangu_vl_7b_bf16_mxfp4_w4a8_smoke_${TOTAL_TRAINING_STEPS}step}
export LOG_DIR=${LOG_DIR:-/home/chentaoxu/data/openpangu-vl-npu/mxfp4-w4a8-smoke-20260921/${EXPERIMENT_NAME}}
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/r_mxfp4_w4a8_smoke}

exec "${SCRIPT_DIR}/run_openpangu_vl_7b_npu_mxfp8_sim.sh" \
    actor_rollout_ref.actor.fsdp_config.qat.enable=False \
    "$@"
