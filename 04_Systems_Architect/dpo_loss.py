"""Systems Architect: DPO loss and Bradley-Terry log-ratio computation, from scratch.

Math (Rafailov et al., 2023):
    r_hat(x, y)  = beta * log( pi_theta(y|x) / pi_ref(y|x) )        implicit reward (log Z cancels)
    P(y_w > y_l) = sigmoid( r_hat(x, y_w) - r_hat(x, y_l) )         Bradley-Terry
    L_DPO        = - E[ log sigmoid( beta*(log-ratio_w - log-ratio_l) ) ]
    dL/dtheta    = -beta * sigmoid(r_hat_l - r_hat_w) * ( grad log pi_w - grad log pi_l )
The sigmoid factor is the dynamic weight: large when the model ranks the loser higher, ~0 when confident.
"""
from typing import Dict, NamedTuple

import torch
import torch.nn.functional as F


def sequence_logprobs(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor,
                      average: bool = False) -> torch.Tensor:
    """Sum (or mean) of per-token log-probs of `labels` under `logits`.

    logits: [B, T, V] already aligned so logits[:, t] predicts labels[:, t]
    labels: [B, T] token ids; mask: [B, T] 1 for response tokens, 0 for prompt/padding.
    """
    logp = torch.log_softmax(logits.float(), dim=-1)
    tok = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1) * mask
    total = tok.sum(-1)
    return total / mask.sum(-1).clamp(min=1) if average else total


class DPOOutput(NamedTuple):
    loss: torch.Tensor
    reward_chosen: torch.Tensor
    reward_rejected: torch.Tensor
    metrics: Dict[str, float]


def log_ratios(policy_logps: torch.Tensor, ref_logps: torch.Tensor) -> torch.Tensor:
    """log pi_theta(y|x) - log pi_ref(y|x) for whole sequences (Bradley-Terry building block)."""
    return policy_logps - ref_logps


def dpo_loss(policy_chosen_logps: torch.Tensor, policy_rejected_logps: torch.Tensor,
             ref_chosen_logps: torch.Tensor, ref_rejected_logps: torch.Tensor,
             beta: float = 0.1, label_smoothing: float = 0.0) -> DPOOutput:
    """Standard DPO loss on sequence-level log-probs (each tensor shape [B])."""
    if beta <= 0:
        raise ValueError("beta must be > 0")
    if not 0.0 <= label_smoothing < 0.5:
        raise ValueError("label_smoothing must be in [0, 0.5)")
    chosen_lr = log_ratios(policy_chosen_logps, ref_chosen_logps)
    rejected_lr = log_ratios(policy_rejected_logps, ref_rejected_logps)
    margin = beta * (chosen_lr - rejected_lr)          # r_hat_w - r_hat_l
    loss = -(1 - label_smoothing) * F.logsigmoid(margin) - label_smoothing * F.logsigmoid(-margin)
    rc, rr = (beta * chosen_lr).detach(), (beta * rejected_lr).detach()
    metrics = {
        "loss": loss.mean().item(),
        "reward_accuracy": (rc > rr).float().mean().item(),
        "reward_margin": (rc - rr).mean().item(),
        "reward_chosen": rc.mean().item(),
        "reward_rejected": rr.mean().item(),
    }
    return DPOOutput(loss.mean(), rc, rr, metrics)


def dpo_gradient_weight(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps,
                        ref_rejected_logps, beta: float = 0.1) -> torch.Tensor:
    """beta * sigmoid(r_hat_l - r_hat_w): the per-pair weight in the DPO gradient."""
    margin = beta * ((policy_chosen_logps - ref_chosen_logps) - (policy_rejected_logps - ref_rejected_logps))
    return beta * torch.sigmoid(-margin)
