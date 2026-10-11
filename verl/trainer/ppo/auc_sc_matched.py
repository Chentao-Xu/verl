"""Center the full PPO field or its mismatch residual relative to the old policy.
Both use the actual vanilla/dual-clipped derivative and a fixed advantage sign.
Full-vocabulary exact; sparse head uses joint q/old-policy tail reconstruction.
"""
import math
import torch
from verl.trainer.ppo.core_algos import agg_loss


def effective_score(logp, old_logp, advantages, weight, config):
    """Detached -d(PPO token loss)/dlogp divided by A; includes numerical clamp.

    Equality at dual clipping uses torch.minimum's half subgradient. At regular
    clip boundaries both matching branches differentiate, so the gate is one.
    """
    with torch.no_grad():
        lo = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
        hi = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio
        dual = config.get('clip_ratio_c', 3.)
        d = logp - old_logp
        r = d.clamp(-20., 20.).exp()
        neg_gate = (r >= 1-lo).to(r) * ((r < dual).to(r) + .5*(r == dual).to(r))
        gate = torch.where(advantages > 0, (r <= 1+hi).to(r),
                           torch.where(advantages < 0, neg_gate, torch.zeros_like(r)))
        gate *= ((d >= -20.) & (d <= 20.)).to(r)
        return weight*r*gate, gate


def matched_correction(*, trainer_head_logp, old_head_logp, action_log_probs,
                       advantages, response_mask, topk_ids, topk_log_probs,
                       topk_mask, loss_agg_mode, config, weight_fn, eps=1e-6,
                       sampler_action_logp=None, allocation_lambda=0.0, target="full", return_token_loss=False):
    if target not in ("full", "mismatch"):
        raise ValueError('SC target must be full or mismatch')
    if target == "mismatch" and allocation_lambda != 0:
        raise ValueError('mismatch SC requires allocation lambda=0')
    # Full-field mode optionally reallocates its center. Mismatch mode keeps
    # the original common residual mean and forbids that separate proposal.
    if not math.isfinite(allocation_lambda) or not 0 <= allocation_lambda <= 1:
        raise ValueError('SC allocation lambda must be finite and in [0,1]')
    if allocation_lambda and (sampler_action_logp is None or sampler_action_logp.shape != action_log_probs.shape):
        raise ValueError('weighted SC requires sampled rollout logprobs')
    if old_head_logp is None:
        raise ValueError('matched SC requires old-policy head logprobs from the pre-update forward')
    if any(t.shape != trainer_head_logp.shape for t in (old_head_logp,topk_ids,topk_log_probs,topk_mask)):
        raise ValueError('matched SC head shapes disagree')
    if trainer_head_logp.shape[:2] != action_log_probs.shape or trainer_head_logp.ndim != 3:
        raise ValueError('matched SC response shape mismatch')
    if advantages.shape != action_log_probs.shape or response_mask.shape != advantages.shape:
        raise ValueError('matched SC advantage/mask shape mismatch')
    if topk_mask.dtype != torch.bool:raise TypeError('SC head mask must be boolean')
    active=response_mask.bool();mask=topk_mask & active.unsqueeze(-1)
    if bool((active & ~mask.any(-1)).any()):raise ValueError('empty head at active position')
    if any(not bool(torch.isfinite(t[mask]).all()) for t in (trainer_head_logp,old_head_logp,topk_log_probs)):
        raise ValueError('nonfinite matched SC evidence')
    if not bool(torch.isfinite(advantages[active]).all()) or bool((topk_ids[mask]<0).any()):raise ValueError('invalid SC inputs')
    ids=topk_ids.masked_fill(~mask,-1).sort(-1).values
    if bool(((ids[...,1:]==ids[...,:-1]) & (ids[...,1:]>=0)).any()):raise ValueError('duplicate fixed head IDs')
    dtype=torch.float64 if trainer_head_logp.dtype==torch.float64 else torch.float32
    lp=trainer_head_logp.to(dtype).masked_fill(~mask,0)
    with torch.no_grad():
        lq=topk_log_probs.to(dtype).masked_fill(~mask,0)
        old=old_head_logp.to(dtype).masked_fill(~mask,0)
        p=lp.exp().masked_fill(~mask,0);q=lq.exp().masked_fill(~mask,0);o=old.exp().masked_fill(~mask,0)
        pm,qm,om=p.sum(-1),q.sum(-1),o.sum(-1)
        if bool(((pm>1+2e-5)|(qm>1+2e-5)|(om>1+2e-5)).any()):raise ValueError('SC head mass exceeds one')
        pt=(1-pm).clamp_min(eps);qt=(1-qm).clamp_min(0);ot=(1-om).clamp_min(eps)
        a=advantages.detach().to(dtype).masked_fill(~active,0)
        anchor=old if target == "mismatch" else lp
        anchor_tail=ot if target == "mismatch" else pt
        head_w=weight_fn((anchor-lq).exp())
        tail_w=weight_fn(anchor_tail/qt.clamp_min(eps))
        head_h,head_gate=effective_score(lp,old,a.unsqueeze(-1),head_w,config)
        tail_h,tail_gate=effective_score(pt.log(),ot.log(),a,tail_w,config)
        residual=(q*head_h-(qt/pt*tail_h).unsqueeze(-1)*p).masked_fill(~mask,0)
        full_residual=residual
        reference_residual=torch.zeros_like(residual)
        if target == "mismatch":
            # mu_err = E_q[w beta s] - E_old[beta s]. Form the
            # signed probability mass directly, avoiding huge old/q ratios.
            head_beta,_=effective_score(lp,old,a.unsqueeze(-1),torch.ones_like(p),config)
            tail_beta,_=effective_score(pt.log(),ot.log(),a,torch.ones_like(pt),config)
            old_tail_mass=(1-om).clamp_min(0)
            reference_residual=(o*head_beta-(old_tail_mass/pt*tail_beta).unsqueeze(-1)*p).masked_fill(~mask,0)
            error_head=(q*head_w-o)*head_beta
            error_tail=(qt*tail_w-old_tail_mass)*tail_beta
            residual=(error_head-(error_tail/pt).unsqueeze(-1)*p).masked_fill(~mask,0)
        z=(q*head_w).sum(-1)+qt*tail_w
        allocation=torch.ones_like(a)
        if allocation_lambda:
            if not bool(torch.isfinite(action_log_probs[active]).all()) or not bool(torch.isfinite(sampler_action_logp[active]).all()):
                raise ValueError('nonfinite sampled logprobs for weighted SC')
            sample_logr=(action_log_probs.detach().to(dtype).masked_fill(~active,0)
                         -sampler_action_logp.detach().to(dtype).masked_fill(~active,0))
            sample_w=weight_fn(sample_logr.exp())
            # No floor: silently flooring Z would violate E[k]=1. Reject a
            # degenerate normalizer; masked/padded tokens use a benign value.
            if not bool(torch.isfinite(z[active]).all()) or bool((z[active] <= 0).any()):
                raise ValueError('weighted SC requires positive finite per-prefix weight mass Z')
            allocation=1-allocation_lambda+allocation_lambda*sample_w/z.masked_fill(~active,1)
            if not bool(torch.isfinite(allocation[active]).all()):
                raise ValueError('nonfinite SC allocation coefficient')
            residual=residual*allocation.unsqueeze(-1)
        stats={'actor/sc_matched':1.,'actor/sc_old_head_mass':float(om[active].mean()) if active.any() else 0.,
               'actor/sc_mismatch_only':float(target == "mismatch"),
               'actor/sc_full_center_coeff_abs_mean':float(full_residual.abs().sum(-1)[active].mean()) if active.any() else 0.,
               'actor/sc_preserved_ppo_coeff_abs_mean':float(reference_residual.abs().sum(-1)[active].mean()) if active.any() else 0.,
               'actor/sc_tail_q_mass':float(qt[active].mean()) if active.any() else 0.,
               'actor/sc_head_active_q_mass':float((q*head_gate).sum(-1)[active].mean()) if active.any() else 0.,
               'actor/sc_tail_gate_mean':float(tail_gate[active].mean()) if active.any() else 0.,
               'actor/sc_coefficient_abs_mean':float(residual.abs().sum(-1)[active].mean()) if active.any() else 0.,
               'actor/sc_allocation_lambda':float(allocation_lambda),
               'actor/sc_weight_mass_mean':float(z[active].mean()) if active.any() else 0.,
               'actor/sc_weight_mass_min':float(z[active].min()) if active.any() else 0.,
               'actor/sc_allocation_mean':float(allocation[active].mean()) if active.any() else 0.,
               'actor/sc_allocation_max':float(allocation[active].max()) if active.any() else 0.}
    loss_mat=a*(residual*lp).sum(-1)
    if return_token_loss:return loss_mat,qm[active].mean(),stats
    if not bool(active.any()):return loss_mat.sum(),qm.sum()*0,stats
    if loss_agg_mode=='seq-mean-token-mean':
        loss=((loss_mat*response_mask).sum(-1)/response_mask.sum(-1).clamp_min(1)).mean()
    else:loss=agg_loss(loss_mat=loss_mat,loss_mask=response_mask,loss_agg_mode=loss_agg_mode)
    return loss,qm[active].mean(),stats
