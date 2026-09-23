"""Top-k score centering for rollout/trainer policy mismatch.

The sampler distribution is represented by processed top-k logprobs; its tail
uses the trainer distribution rescaled to match the missing sampler mass.
"""

import torch


def correction_for_response_logits(
    logits: torch.Tensor, micro_batch
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align flattened full-sequence logits with next-token response top-k data."""
    seq_offsets = micro_batch["input_ids"].offsets()
    responses = micro_batch["responses"]
    if responses.is_nested:
        response_lens = responses.offsets().diff()
    else:
        prompt_width = micro_batch["prompts"].shape[1]
        response_lens = micro_batch["attention_mask"][:, prompt_width:].sum(dim=1)
    positions = torch.cat(
        [
            torch.arange(
                seq_offsets[i + 1] - response_lens[i] - 1,
                seq_offsets[i + 1] - 1,
                device=logits.device,
            )
            for i in range(len(response_lens))
        ]
    ).long()
    ids = torch.cat(
        [micro_batch["score_centering_ids"][i, :response_lens[i]] for i in range(len(response_lens))]
    ).to(logits.device)
    q_logprobs = torch.cat(
        [micro_batch["score_centering_logprobs"][i, :response_lens[i]] for i in range(len(response_lens))]
    ).to(logits.device)
    correction, q_mass, p_mass = topk_score_correction(logits.index_select(0, positions), ids, q_logprobs)
    full_correction = torch.zeros(logits.shape[0], device=logits.device, dtype=correction.dtype)
    return full_correction.index_copy(0, positions, correction), q_mass.mean(), p_mass.mean()


def topk_score_correction(
    trainer_logits: torch.Tensor,
    sampler_topk_ids: torch.Tensor,
    sampler_topk_logprobs: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return differentiable correction, sampler head mass and trainer head mass.

    Inputs have shapes [tokens, vocab], [tokens, k], [tokens, k]. Invalid
    top-k slots have logprob -inf and contribute zero mass.
    """
    if trainer_logits.ndim != 2 or sampler_topk_ids.shape != sampler_topk_logprobs.shape:
        raise ValueError("Invalid score-centering top-k tensor shapes")
    if sampler_topk_ids.shape[0] != trainer_logits.shape[0]:
        raise ValueError("Score-centering top-k data are not token-aligned")
    if not torch.isfinite(trainer_logits).all():
        raise ValueError("Non-finite actor logits during score centering")

    logits = trainer_logits.float()
    head_logp = logits.gather(-1, sampler_topk_ids.long()) - torch.logsumexp(logits, dim=-1, keepdim=True)
    sampler_prob = sampler_topk_logprobs.float().exp()
    trainer_prob = head_logp.exp()
    sampler_mass = sampler_prob.sum(-1)
    trainer_mass = trainer_prob.sum(-1)
    if (sampler_mass > 1.0001).any() or (sampler_mass <= 0).any():
        raise ValueError("Sampler top-k probabilities are invalid or not normalized")
    rho = (1 - sampler_mass).clamp_min(eps) / (1 - trainer_mass).clamp_min(eps)
    residual = (sampler_prob - rho.unsqueeze(-1) * trainer_prob).detach()
    correction = (residual * head_logp).sum(-1)
    return correction, sampler_mass.detach(), trainer_mass.detach()
