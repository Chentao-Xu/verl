"""Engine bridge for fixed rollout head and PPO-matched residual centering."""
import math
import torch
from verl.trainer.ppo.auc_sc_matched import matched_correction
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn

def head_for_response_logits(logits, data):
    offsets = data['input_ids'].offsets()
    responses = data['responses']
    lens = (responses.offsets().diff() if responses.is_nested else
            data['attention_mask'][:, data['prompts'].shape[1]:].sum(-1))
    positions = torch.cat([torch.arange(offsets[i+1]-n-1, offsets[i+1]-1, device=logits.device)
                           for i,n in enumerate(lens)]).long()
    ids = torch.cat([data['score_centering_ids'][i,:n] for i,n in enumerate(lens)]).long()
    # Chunked differentiable full-vocabulary normalization; never head-only softmax.
    from torch.utils.checkpoint import checkpoint
    def gather(x, idx):
        x = x.float()
        return x.gather(-1, idx) - x.logsumexp(-1, keepdim=True)
    parts = []
    for start in range(0, positions.numel(), 64):
        x = logits.index_select(0, positions[start:start+64])
        idx = ids[start:start+64]
        parts.append(checkpoint(gather, x, idx, use_reentrant=False) if torch.is_grad_enabled() else gather(x, idx))
    head = torch.cat(parts)
    full = head.new_zeros((logits.shape[0], ids.shape[-1])).index_copy(0, positions, head)
    return torch.nested.nested_tensor_from_jagged(full, offsets)

def combined_loss(config, logp, head, data):
    mask = data['response_mask'].bool()
    old = data['old_log_probs'].detach()
    q = data['rollout_log_probs'].detach()
    advantages = data['advantages'].detach()
    alpha_values = data['auc_sc_alpha'][mask].detach()
    if not alpha_values.numel() or not torch.isfinite(alpha_values).all() or not (alpha_values == alpha_values[0]).all():
        raise ValueError('Nonconstant/nonfinite batch alpha')
    alpha = alpha_values[0]
    def weight(ratio):
        return ratio.clamp(.2,5.).pow(alpha)
    with torch.no_grad():
        w = ((old-q).masked_fill(~mask,0).clamp(-math.log(5.),math.log(5.))*alpha).exp()
    ppo, metrics = get_policy_loss_fn('vanilla')(
        old_log_prob=old, log_prob=logp, advantages=advantages*w,
        response_mask=mask, loss_agg_mode=config.loss_agg_mode, config=config)
    valid_head = torch.isfinite(data['score_centering_logprobs']) & mask.unsqueeze(-1)
    if (valid_head[mask].sum(-1) != 128).any():
        raise ValueError('Incomplete fixed top128 evidence')
    token_sc, _, diagnostics = matched_correction(
        trainer_head_logp=head, old_head_logp=data['old_policy_head_log_probs'].detach(),
        action_log_probs=logp, advantages=advantages, response_mask=mask,
        topk_ids=data['score_centering_ids'], topk_log_probs=data['score_centering_logprobs'].detach(),
        topk_mask=valid_head, loss_agg_mode=config.loss_agg_mode, config=config,
        weight_fn=weight, sampler_action_logp=q, allocation_lambda=0., target='mismatch', return_token_loss=True)
    correction = agg_loss(token_sc, mask, config.loss_agg_mode, **config.global_batch_info)
    metrics.update(diagnostics)
    metrics.update({'actor/sc_power_alpha':float(alpha), 'actor/sc_pg_scale_enabled':0., 'actor/sc_mismatch_aux_loss_enabled':0.})
    return ppo+correction, metrics, ppo, correction
