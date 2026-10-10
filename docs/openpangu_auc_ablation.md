# OpenPangu AUC W4A4 ablation (2026-10-10)

User-selected variant: token-mean actor loss, mismatch auxiliary loss disabled.
This is NOT an exact reproduction of the historical seq-mean + mismatch method.

## Provenance

`verl/trainer/ppo/auc_controller.py` and `auc_profile.json` are vendored without
modification from the supplied `bilateral_mismatch_pg_handoff/bilateral_control`
package. Controller SHA256:
`ccdeca0a5d0156d3ae651bd0bf4860c6d6628e866b8434c990c605619a8a00ec`.
The handoff did not establish an exact upstream commit. Its original archive SHA256
is `91058162320a29993e5f1bb166e4f82ab5185b8059d6424da17837b864912261`.

## Integration

Prepare once per full driver batch after GRPO advantages. CPU FP32 controller
consumes detached old actor logprob, real quantized sampler logprob, advantages,
and aligned response mask. It emits detached symmetric-power weights (cap 5)
and a global PG scale. Historical risk statistics remain sequence-averaged.
Actor applies weights to advantages exactly once, uses existing dual-clipped PPO
with token-mean/global token normalization, then scales PG only. Reference KL
(.01, low_var_kl) is added unscaled. No SC, ordinary TIS/RS, mismatch SmoothL1,
or auxiliary autograd calls. Historical beta remains diagnostic, NOT a loss
coefficient in this variant. PG gate remains active after the calibration window.

State commits only after reported successful finite BF16 optimizer update;
failure aborts and requires restoring model/optimizer/controller together.
Each checkpoint saves `auc_state.json` before publishing the latest-step marker.
Resume rejects missing/incompatible controller state. Current adapter requires
one PPO epoch and one full-batch optimizer update per rollout batch.

## Launcher and comparison

`examples/grpo_trainer/run_openpangu_vl_7b_npu_bf16_hif4_w4a4_auc.sh` retains
the SC baseline's 4K dataset, batch64, n8, mini64, response1024, prompt7168,
dynamic4096, concurrency32, LR1e-6, KL.01, frozen BF16 vision tower, BF16 actor,
HiF4 W4A4 language 0--33 rollout QDQ, and safe SHM weight sync.
Formal defaults: 930 updates/15 epochs, eval25, save100, all checkpoints retained.
Formal training requires separate user confirmation.

Single-step acceptance overrides total steps=1, initial/every-step evaluation off,
console-only logging, save frequency=1 (tests controller checkpoint persistence).
An actual one-step pass does not validate long-run convergence or the step>50
controller gate; the latter is covered synthetically in CPU tests.

CPU tests: `python3 -m pytest -q tests/workers/utils/test_auc_adapter.py`.
AI-assisted implementation; human review required before upstream publication.
