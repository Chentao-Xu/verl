# OpenPangu-VL-7B: MXFP4 rollout score centering

Scope: BF16 Actor, simulated MXFP4-weight/MXFP8-activation rollout, no Actor QAT, no TIS. This branch is a **short-link adaptation**, not an authorized convergence experiment.

## Algorithm and data path

The vLLM sampler returns processed top-128 log-probabilities plus the sampled-token log-probability. The latter is already used by Verl; the top-k pairs are carried with each single-turn response into the Actor batch. The Actor computes the detached head residual

`q_v - rho * p_v`, where `rho = (1 - sum_H q) / (1 - sum_H p)`,

and optimizes `-A * (log p(sampled) - sum_H stopgrad(q_v - rho*p_v) * log p_v)`. GRPO advantages and the existing reference KL coefficient 0.01 remain unchanged. The PPO clip term is replaced by this REINFORCE-style centered term. The source is [Score Centering Stabilizes Off-policy Reinforcement Learning, Eq. 8](https://arxiv.org/html/2609.20807v1).

The initial implementation requires the existing FSDP no-SP, non-fused Actor logit path and single-turn rollout. It rejects unsupported fused/SP paths and missing top-k data instead of silently falling back. The sampled token need not belong to the top-128 head.

## One-step acceptance test

Entry point: `examples/grpo_trainer/run_openpangu_vl_7b_npu_bf16_mxfp4_score_centering_smoke.sh`. It runs one step on 8 NPUs with batch 8 and rollout `n=2`, response length 256, `score_centering_topk=128`, TIS disabled, console-only logging, no validation/checkpoint/WandB. It uses the fixed ViRL4K train subset as a source of smoke-test examples, not a convergence run.

Numerical tests: `tests/workers/utils/test_score_centering.py` verify constant-reward expected-gradient cancellation, invalid top-k slot masking, and alignment with padded OpenPangu responses.

Logs on machine 1: `/home/chentaoxu/data/openpangu-eval-work/training/score-centering-smoke-20260923/`. The first attempt reached Actor update and exposed padded response alignment; the second stopped before Ray startup because its Unix socket path was too long. The third reached Actor backward but exposed a scalar-output incompatibility in Verl's postprocessor. Both code issues were fixed. The final attempt is `retry4/console.log` and exited with code 0.

Final one-step results (2026-09-23): `training/global_step=1`, `actor/score_centering/sampler_head_mass=0.999658`, `actor/loss=0.019411`, `actor/grad_norm=2.275259`, and `timing_s/step=162.92`. The log confirms MXFP4 simulation quantized 36 language weight tensors per reload and marked 136 language Linear layers for MXFP8 activation QDQ. All eight NPUs were idle after exit. Numerical tests passed 3/3 in the existing training container.

This proves the short training chain, not convergence or 4-bit stability. Do not start formal 4K/6-epoch training before user approval.
