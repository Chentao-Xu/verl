import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict

from verl.trainer.ppo import auc_adapter as auc
from verl.trainer.ppo.auc_controller import compute_exploration_tis
from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla, kl_penalty
from verl.utils import tensordict_utils as tu
from verl.workers.config.actor import ActorConfig, PolicyLossConfig
from verl.workers.utils import losses


def batch():
    return TensorDict({
        "old_log_probs": torch.tensor([[-1., -2., 0.], [-1., -1., -1.]]),
        "rollout_log_probs": torch.tensor([[-2., -1., 0.], [-2., -2., -2.]]),
        "advantages": torch.tensor([[1., 1., 0.], [-1., -1., -1.]]),
        "response_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
    }, batch_size=[2])


def test_controller_identity_and_resume(tmp_path):
    b = batch()
    state = auc.ExplorationTISState.create()
    result = auc.prepare(b, state, 1, 3)
    profile = json.loads(Path(auc.__file__).with_name("auc_profile.json").read_text())
    expected = compute_exploration_tis(b["old_log_probs"], b["rollout_log_probs"], b["advantages"],
                                      b["response_mask"], state=state, source_step=1, response_length_cap=3, **profile)
    torch.testing.assert_close(result.weights, expected.weights, rtol=0, atol=0)
    assert state.source_step == 0
    with pytest.raises(RuntimeError):
        auc.commit(result, {"actor/optimizer_step_succeeded": [1., 0.], "actor/grad_norm": [1., 1.]})
    state = auc.commit(result, {"actor/optimizer_step_succeeded": [1., 1.], "actor/grad_norm": [1., 1.]})
    auc.save_state(tmp_path, state, 1)
    restored = auc.load_state(tmp_path, 1)
    assert restored == state
    a = auc.prepare(batch(), state, 2, 3)
    b = auc.prepare(batch(), restored, 2, 3)
    assert a.proposed_state == b.proposed_state
    torch.testing.assert_close(a.weights, b.weights, rtol=0, atol=0)


@pytest.mark.parametrize("key", ["old_log_probs", "rollout_log_probs", "advantages"])
def test_reject_bad_valid_token(key):
    b = batch()
    b[key][0, 0] = float("nan")
    with pytest.raises(ValueError):
        auc.prepare(b, auc.ExplorationTISState.create(), 1, 3)


def test_pg_gate_remains_enabled_without_aux_loss():
    state = replace(auc.ExplorationTISState.create(), source_step=57,
                    mismatch_baseline_count=50, mismatch_baseline_ema=0.01,
                    mismatch_ema=10., mismatch_ess_baseline_ema=1.,
                    mismatch_support_baseline_ema=1., mismatch_excess_steps=7)
    result = auc.prepare(batch(), state, 58, 3)
    assert result.metrics[auc.PREFIX + "mismatch_trust_pg_scale"] < 1
    assert result.metrics["auc/mismatch_effective_coef"] == 0


def test_actual_ppo_path_token_mean_kl_and_microbatch(monkeypatch):
    # Only bypass output unpacking here; exercise actual ppo_loss/global normalization.
    monkeypatch.setattr(losses, "no_padding_2_padding", lambda x, _: x)
    b = batch()
    auc.prepare(b, auc.ExplorationTISState.create(), 1, 3)
    b["auc_pg_scale"].fill_(0.3)
    b["ref_log_prob"] = b["old_log_probs"] - 0.1
    current = (b["old_log_probs"] + 0.05).clone().requires_grad_()
    cfg = ActorConfig(strategy="fsdp2", ppo_micro_batch_size_per_gpu=1, rollout_n=8,
                      policy_loss=PolicyLossConfig(loss_mode="auc"),
                      use_kl_loss=True, kl_loss_coef=.01)
    tu.assign_non_tensor(b, dp_size=1, batch_num_tokens=5, global_batch_size=2)
    actual, _ = losses.ppo_loss(cfg, {"log_probs": current}, b)
    pg, _ = compute_policy_loss_vanilla(b["old_log_probs"], current,
                                       b["advantages"] * b["auc_weights"],
                                       b["response_mask"].bool(), config=cfg)
    kl = (kl_penalty(current, b["ref_log_prob"], "low_var_kl") * b["response_mask"]).sum() / 5
    expected = .3 * pg + .01 * kl
    torch.testing.assert_close(actual, expected)
    grad = torch.autograd.grad(actual, current, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, current, retain_graph=True)[0]
    torch.testing.assert_close(grad, expected_grad)
    # Reordered microbatches preserve weights/scales and full-batch token denominator.
    split = sum(losses.ppo_loss(cfg, {"log_probs": current[i:i+1]}, b[i:i+1])[0] for i in (1, 0))
    torch.testing.assert_close(split, actual)
    torch.testing.assert_close(torch.autograd.grad(split, current)[0], grad)
    # Turning on mismatch is explicitly unsupported, rather than silently ignored.
    cfg = replace(cfg, policy_loss=replace(cfg.policy_loss, auc_mismatch_loss=True))
    with pytest.raises(ValueError):
        losses.ppo_loss(cfg, {"log_probs": current}, b)
