"""OpenPangu AUC ablation: historical controller, token-mean, no mismatch loss.

Controller/profile are vendored byte-for-byte from the 2026-09-30 handoff.
Risk statistics retain historical sequence means; actor objective uses token means.
"""

import json
import math
from pathlib import Path

import torch

from .auc_controller import ExplorationTISState, compute_exploration_tis

PREFIX = "rollout_corr/exploration_tis_"


def validate_config(config):
    actor = config.actor_rollout_ref.actor
    rollout = config.actor_rollout_ref.rollout
    correction = config.algorithm.get("rollout_correction") or {}
    if actor.loss_agg_mode != "token-mean" or actor.policy_loss.get("auc_mismatch_loss", False):
        raise ValueError("This AUC adapter requires token-mean and auc_mismatch_loss=False")
    if correction.get("rollout_is") or correction.get("rollout_rs") or correction.get("bypass_mode", False):
        raise ValueError("AUC cannot be combined with ordinary TIS/RS/bypass")
    if rollout.get("score_centering_topk", 0) or not rollout.calculate_log_probs:
        raise ValueError("AUC requires sampled rollout logprobs, without SC top-k")
    if rollout.temperature != 1.0 or rollout.top_p != 1.0 or rollout.top_k != -1:
        raise ValueError("AUC requires temperature=1, top_p=1, top_k=-1")
    if rollout.logprobs_mode != "raw_logprobs":
        raise ValueError("AUC expects raw sampled-token logprobs")
    if actor.ppo_epochs != 1 or actor.ppo_mini_batch_size != config.data.train_batch_size:
        raise ValueError("This adapter requires one optimizer update per rollout batch")


def prepare(batch, state, step, response_length_cap):
    if "rollout_is_weights" in batch:
        raise ValueError("Duplicate rollout correction weights")
    mask = batch["response_mask"].detach().cpu()
    if not torch.all((mask == 0) | (mask == 1)) or not mask.bool().any(-1).all():
        raise ValueError("AUC requires binary masks with nonempty responses")
    valid = mask.bool()
    inputs = []
    for key in ("old_log_probs", "rollout_log_probs", "advantages"):
        value = batch[key].detach().to(device="cpu", dtype=torch.float32)
        if value.shape != mask.shape or not torch.isfinite(value[valid]).all():
            raise ValueError(f"Invalid AUC input: {key}")
        inputs.append(torch.where(valid, value, 0.0))
    profile = json.loads(Path(__file__).with_name("auc_profile.json").read_text())
    result = compute_exploration_tis(
        *inputs, mask, state=state, source_step=step, response_length_cap=response_length_cap, **profile
    )
    if not result.actor_update_allowed or result.metrics[PREFIX + "state_held"]:
        raise RuntimeError("AUC controller rejected this update")
    scale = result.metrics[PREFIX + "mismatch_trust_pg_scale"]
    batch["auc_weights"] = result.weights.to(batch["advantages"].device)
    # Response-shaped constant keeps all engine packing/shuffling paths aligned.
    batch["auc_pg_scale"] = torch.full_like(batch["auc_weights"], scale)
    result.metrics["auc/mismatch_loss_enabled"] = 0.0
    result.metrics["auc/mismatch_effective_coef"] = 0.0
    return result


def commit(result, metrics):
    def flatten(value):
        if isinstance(value, (list, tuple)):
            return [x for item in value for x in flatten(item)]
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().flatten().tolist()
        return [float(value)]

    success = flatten(metrics.get("actor/optimizer_step_succeeded", 0))
    norms = flatten(metrics.get("actor/grad_norm", float("nan")))
    if not all(v == 1 for v in success) or not all(math.isfinite(v) for v in norms):
        raise RuntimeError("AUC optimizer update failed; controller not committed; restore last checkpoint")
    return result.proposed_state


def save_state(folder, state, step):
    if state.source_step != step:
        raise RuntimeError("AUC/controller checkpoint step mismatch")
    target = Path(folder) / "auc_state.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"version": 1, "variant": "token-mean-no-mismatch", "state": state.state_dict()}, indent=2))
    temporary.replace(target)


def load_state(folder, step):
    value = json.loads((Path(folder) / "auc_state.json").read_text())
    if value["version"] != 1 or value["variant"] != "token-mean-no-mismatch":
        raise ValueError("Incompatible AUC checkpoint")
    state = ExplorationTISState.from_state_dict(value["state"])
    if state.source_step != step:
        raise ValueError("AUC/controller restore step mismatch")
    return state
