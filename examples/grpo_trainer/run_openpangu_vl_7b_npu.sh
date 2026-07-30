#!/usr/bin/env bash
# GRPO | OpenPangu-VL-7B | vLLM rollout | FSDP2 training | Ascend NPU

set -euo pipefail
set -x

MODEL_PATH=${MODEL_PATH:-/data/models/openPangu-VL-7B}
TRAIN_FILE=${TRAIN_FILE:-/data/datasets/geo3k/train.parquet}
TEST_FILE=${TEST_FILE:-/data/datasets/geo3k/test.parquet}

NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-8}
ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_N=${ROLLOUT_N:-2}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.45}
UPDATE_WEIGHTS_BUCKET_MEGABYTES=${UPDATE_WEIGHTS_BUCKET_MEGABYTES:-2048}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1536}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}
MAX_MODEL_LENGTH=${MAX_MODEL_LENGTH:-2048}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-4096}
ACTOR_ATTN_IMPLEMENTATION=${ACTOR_ATTN_IMPLEMENTATION:-sdpa}
FSDP_CPU_OFFLOAD_POLICY=${FSDP_CPU_OFFLOAD_POLICY:-False}

if ((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH > MAX_MODEL_LENGTH)); then
    echo "MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH must not exceed MAX_MODEL_LENGTH" >&2
    exit 2
fi

TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-1}
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:--1}
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-8}
ROLLOUT_ONLY=${ROLLOUT_ONLY:-0}
FILTER_OVERLONG_PROMPTS=${FILTER_OVERLONG_PROMPTS:-False}
DATA_LOADER_NUM_WORKERS=${DATA_LOADER_NUM_WORKERS:-0}
AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS:-8}

PROJECT_NAME=${PROJECT_NAME:-openpangu_vl_grpo_npu}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-openpangu_vl_7b_fsdp2_vllm}
LOG_DIR=${LOG_DIR:-/data/openpangu-vl-npu/stage5}
LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS:-8}
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-${LOG_DIR}/${EXPERIMENT_NAME}-validation}
TRAINER_LOGGER=${TRAINER_LOGGER:-'["console"]'}
SAVE_FREQ=${SAVE_FREQ:--1}
TEST_FREQ=${TEST_FREQ:--1}
RESUME_MODE=${RESUME_MODE:-disable}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-/data/openpangu-vl-npu/checkpoints/${EXPERIMENT_NAME}}
mkdir -p "${LOG_DIR}"

export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1500}
export HCCL_HOST_SOCKET_PORT_RANGE=${HCCL_HOST_SOCKET_PORT_RANGE:-60000-60050}
export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-61000-61050}
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export VERL_LOGGING_LEVEL=${VERL_LOGGING_LEVEL:-WARN}
export VERL_WEIGHT_SYNC_CHECKSUM=${VERL_WEIGHT_SYNC_CHECKSUM:-0}
export VERL_ROLLOUT_LOGPROB_CHECK=${VERL_ROLLOUT_LOGPROB_CHECK:-0}

DATA=(
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.image_key=images
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.train_max_samples=${TRAIN_MAX_SAMPLES}
    data.val_max_samples=${VAL_MAX_SAMPLES}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=${FILTER_OVERLONG_PROMPTS}
    data.filter_overlong_prompts_workers=1
    data.dataloader_num_workers=${DATA_LOADER_NUM_WORKERS}
    data.truncation=error
    data.shuffle=False
    data.trust_remote_code=True
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.trust_remote_code=True
    +actor_rollout_ref.model.override_config.attn_implementation="${ACTOR_ATTN_IMPLEMENTATION}"
    actor_rollout_ref.model.use_remove_padding=False
    actor_rollout_ref.model.use_fused_kernels=False
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.freeze_vision_tower=True
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${NDEVICES_PER_NODE}
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True
    actor_rollout_ref.actor.fsdp_config.offload_policy=${FSDP_CPU_OFFLOAD_POLICY}
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.dtype=bfloat16
    actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LENGTH}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.enable_chunked_prefill=False
    actor_rollout_ref.rollout.enable_prefix_caching=False
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.max_num_seqs=8
    actor_rollout_ref.rollout.max_num_batched_tokens=4096
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.agent.num_workers=${AGENT_NUM_WORKERS}
    actor_rollout_ref.rollout.checkpoint_engine.backend=naive
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=${UPDATE_WEIGHTS_BUCKET_MEGABYTES}
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0
    +actor_rollout_ref.rollout.engine_kwargs.vllm.limit_mm_per_prompt.image=1
    +actor_rollout_ref.rollout.engine_kwargs.vllm.limit_mm_per_prompt.video=0
)

TRAINER=(
    trainer.logger="${TRAINER_LOGGER}"
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.n_gpus_per_node=${NDEVICES_PER_NODE}
    trainer.nnodes=1
    trainer.resume_mode="${RESUME_MODE}"
    trainer.val_before_train=True
    trainer.val_only=False
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=1
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS}
    trainer.log_val_generations=${LOG_VAL_GENERATIONS}
    trainer.validation_data_dir="${VALIDATION_DATA_DIR}"
    trainer.default_local_dir="${CHECKPOINT_DIR}"
)

if [[ "${ROLLOUT_ONLY}" == "1" ]]; then
    TRAINER+=(trainer.val_only=True)
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "$@" 2>&1 | tee "${LOG_DIR}/${EXPERIMENT_NAME}.log"
