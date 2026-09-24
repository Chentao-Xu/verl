# ViRL4K BF16–MXFP4 W4A8 + Score Centering, 200 steps

Started 2026-09-23 on machine 1 (`liteserver-c009-5-3`) in container `openpangu-grpo-200`. The training code and script were committed before launch at `234f9c145` on branch `feat/openpangu-score-centering`; the worktree was clean. This run was requested as a 200-step experiment, not a one-step smoke test.

- Entry: `examples/grpo_trainer/run_openpangu_vl_7b_npu_bf16_mxfp4_score_centering_200step.sh`
- Run directory: `/home/chentaoxu/data/openpangu-eval-work/training/virl4k-bf16-mxfp4-score-centering-20260923/openpangu_vl_7b_virl4k_bf16_mxfp4_w4a8_sc128_b64n8_s200_seed42_accel_dynbsz4096_conc32/`
- WandB: <https://wandb.ai/ct-xu-hitsz/openpangu_vl_virl4k_grpo/runs/ac7lhz25>
- 4K train subset; fixed 1698-example test set. Batch 64, rollout n=8, response length 1024, 8 NPUs, seed 42, lr 1e-6, dynamic token budget 4096, rollout concurrency 32. Full validation every 25 steps (and before training), checkpoint every 100 steps. These match the preceding BF16–MXFP4 run except score centering top-k 128, TIS disabled, and total steps limited to 200. BF16 Actor, no QAT; simulated MXFP4-weight/MXFP8-activation language rollout.

At first-step acceptance, WandB was syncing online and the process remained running. Initial full-validation `acc/mean@1=0.565842`. Step 1: `actor/score_centering/sampler_head_mass=0.884290`, `rollout_probs_diff_mean=0.069940`, `actor/loss=-1.480889`, `actor/grad_norm=0.903533`, `critic/score/mean=0.551953`, `timing_s/step=642.094` seconds. The top-128 head covers only 88.4% of the sampler distribution on average in this first step; monitor this and reward/stability in later steps rather than treating one finite update as convergence evidence.

Rough remaining-time estimate after step 1: 30–39 hours. This brackets the earlier aligned MXFP4 run's roughly 500-second steady steps and the new first step's 642 seconds, plus eight full validations and checkpoint writes. Estimated finish: approximately 2026-09-25 04:30–13:30 China time if performance remains similar and there are no failures. The run writes `exit_status.txt` on completion or failure.

## Six-epoch continuation

At 2026-09-24 23:52 China time, the 200-step run was still active at step 170. The user requested a continuation to six epochs (372 total steps, 62 steps per epoch). `examples/grpo_trainer/continue_openpangu_vl_7b_npu_bf16_mxfp4_score_centering_6epoch.sh` was queued in the same container. It waits for a successful 200-step exit and complete `global_step_200` FSDP checkpoint, checks that all eight NPUs are idle, and only then resumes from that checkpoint. No current training process is interrupted.

The continuation retains the 4K data, fixed 1698-example validation set, batch 64, rollout n=8, BF16 actor, simulated MXFP4 W4A8 rollout, score-centering top-k 128, no TIS, KL coefficient 0.01, full validation every 25 steps, and checkpoints every 100 steps plus the final step. It writes the additional checkpoints into the same checkpoint directory, using a separate log/manifest under `continuation_step200_to372/`. WandB uses run ID `ac7lhz25` with `WANDB_RESUME=must` to append to the existing run. The queue log is `continuation_queue.log`; if the 200-step run fails or the devices are occupied, continuation refuses to launch.
