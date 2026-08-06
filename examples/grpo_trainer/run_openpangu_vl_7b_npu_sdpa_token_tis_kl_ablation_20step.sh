#!/usr/bin/env bash
# Controlled 20-step KL-loss ablation for OpenPangu-VL-7B on ViRL39K.
#
# Resume the same complete step-350 checkpoint and keep every other training
# parameter identical. Run once with KL_LOSS_COEF=0.001 and once with 0.01.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
case "${KL_LOSS_COEF}" in
    0.001)
        KL_TAG=kl0001
        ;;
    0.01)
        KL_TAG=kl001
        ;;
    *)
        echo "KL_LOSS_COEF must be 0.001 or 0.01 for this controlled ablation" >&2
        exit 2
        ;;
esac

export RESUME_FROM_PATH=${RESUME_FROM_PATH:-/home/chentaoxu/data/openpangu-vl-npu/checkpoints/openpangu_vl_7b_virl39k_grpo_200step_sdpa_token_tis/global_step_350}
if [[ ! -f "${RESUME_FROM_PATH}/data.pt" ]]; then
    echo "Complete step-350 checkpoint is unavailable: ${RESUME_FROM_PATH}" >&2
    exit 3
fi

export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-370}
export RESUME_MODE=resume_path
export TEST_FREQ=${TEST_FREQ:-10}
export SAVE_FREQ=${SAVE_FREQ:-10}

export PROJECT_NAME=${PROJECT_NAME:-openpangu_vl_grpo_npu}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-openpangu_vl_7b_virl39k_sdpa_token_tis_${KL_TAG}_s350_s370}
export LOG_DIR=${LOG_DIR:-/home/chentaoxu/data/openpangu-vl-npu/kl-ablation/${KL_TAG}}
export CHECKPOINT_DIR=${CHECKPOINT_DIR:-/home/chentaoxu/data/openpangu-vl-npu/checkpoints/${EXPERIMENT_NAME}}
export VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-${LOG_DIR}/generations}
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/r_pangu_${KL_TAG}}

exec "${SCRIPT_DIR}/run_openpangu_vl_7b_npu_sdpa_token_tis_20step.sh" \
    trainer.resume_from_path="${RESUME_FROM_PATH}" \
    trainer.max_actor_ckpt_to_keep=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}" \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.use_kl_in_reward=False \
    "$@"
