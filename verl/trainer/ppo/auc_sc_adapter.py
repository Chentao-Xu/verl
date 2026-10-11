"""AT-TIS + PPO mismatch-residual SC, with target baseline reference KL unchanged."""
import json
from pathlib import Path
import torch
from .auc_sc_controller import ExplorationTISState, compute_exploration_tis
from .auc_adapter import commit

PREFIX = 'rollout_corr/exploration_tis_'

def validate_config(config):
    a, r = config.actor_rollout_ref.actor, config.actor_rollout_ref.rollout
    correction = config.algorithm.get('rollout_correction') or {}
    if a.loss_agg_mode != 'token-mean' or a.policy_loss.get('auc_mismatch_loss', False):
        raise ValueError('AUC+SC requires token-mean and mismatch auxiliary loss OFF')
    if any(correction.get(k) for k in ('rollout_is', 'rollout_rs', 'bypass_mode')):
        raise ValueError('Duplicate rollout correction')
    if r.score_centering_topk != 128 or not r.calculate_log_probs or r.logprobs_mode != 'raw_logprobs':
        raise ValueError('AUC+SC requires fixed top128 raw rollout evidence')
    if (r.temperature, r.top_p, r.top_k) != (1., 1., -1):
        raise ValueError('Untransformed training distribution required')
    if a.ppo_epochs != 1 or a.ppo_mini_batch_size != config.data.train_batch_size:
        raise ValueError('One optimizer update per rollout batch required')
    if config.actor_rollout_ref.model.use_remove_padding or config.actor_rollout_ref.model.use_fused_kernels:
        raise ValueError('Initial acceptance requires existing padded model forward, unfused logits')

def prepare(batch, state, step, response_length_cap):
    valid = batch['response_mask'].detach().cpu().bool()
    if not valid.any(-1).all() or 'rollout_is_weights' in batch:
        raise ValueError('Invalid mask or duplicate weights')
    values = []
    for key in ('old_log_probs', 'rollout_log_probs', 'advantages'):
        v = batch[key].detach().cpu().float()
        if v.shape != valid.shape or not torch.isfinite(v[valid]).all():
            raise ValueError(f'Invalid controller input {key}')
        values.append(v.masked_fill(~valid, 0))
    result = compute_exploration_tis(*values, valid, state=state, source_step=step,
        response_length_cap=response_length_cap, cap=5., weight_mode='symmetric_power',
        half_life=8., historical_sequence_risk_enabled=True, sequence_history_size=32,
        sequence_min_history=8, sequence_log_scale_floor=.05, sequence_risk_saturation_z=4.,
        length_risk_enabled=True)
    if not result.actor_update_allowed:
        raise RuntimeError('SC controller rejected update')
    alpha = result.metrics[PREFIX+'alpha_applied']
    batch['auc_sc_alpha'] = torch.full_like(batch['advantages'], alpha)
    result.metrics.update({'auc_sc/pg_scale_enabled': 0., 'auc_sc/mismatch_aux_loss_enabled': 0.})
    return result

def save_state(folder, state, step):
    if state.source_step != step:
        raise ValueError('Controller step mismatch')
    path = Path(folder)/'auc_sc_state.json'
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps({'version': 1, 'variant': 'ppo-mismatch-sc-c5-kl001', 'state': state.state_dict()}))
    tmp.replace(path)

def load_state(folder, step):
    d = json.loads((Path(folder)/'auc_sc_state.json').read_text())
    if d['version'] != 1 or d['variant'] != 'ppo-mismatch-sc-c5-kl001':
        raise ValueError('Incompatible controller state')
    state = ExplorationTISState.from_state_dict(d['state'])
    if state.source_step != step:
        raise ValueError('Controller restore step mismatch')
    return state
