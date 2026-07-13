#!/usr/bin/env bash
# GRPO | OpenPangu-VL-7B | vLLM 0.11 rollout | FSDP2 | Ascend NPU

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
MODEL_PATH=${MODEL_PATH:-/data/models/openPangu-VL-7B}
TRAIN_FILE=${TRAIN_FILE:-/data/datasets/geo3k/train.parquet}
TEST_FILE=${TEST_FILE:-/data/datasets/geo3k/test.parquet}

NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-8}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}
ROLLOUT_N=${ROLLOUT_N:-2}
ROLLOUT_TP=${ROLLOUT_TP:-1}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-1}

for path in "$MODEL_PATH" "$TRAIN_FILE" "$TEST_FILE"; do
    if [[ ! -e "$path" ]]; then
        echo "Required path does not exist: $path" >&2
        exit 1
    fi
done

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH="$MODEL_PATH/inference/vllm_ascend:$REPO_ROOT:${PYTHONPATH:-}"
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=fork
export HCCL_CONNECT_TIMEOUT=1500
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$TEST_FILE" \
    data.image_key=images \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.filter_overlong_prompts=False \
    data.truncation=error \
    data.trust_remote_code=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.external_lib=verl.models.transformers.openpangu_vllm \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.override_config._attn_implementation=eager \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
    actor_rollout_ref.rollout.max_model_len=2560 \
    actor_rollout_ref.rollout.max_num_seqs=8 \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.block_size=128 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.balance_batch=True \
    trainer.logger='["console"]' \
    trainer.project_name=openpangu_vl_grpo \
    trainer.experiment_name=openpangu_vl_7b_npu_smoke \
    trainer.n_gpus_per_node="$NDEVICES_PER_NODE" \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
    "$@"
