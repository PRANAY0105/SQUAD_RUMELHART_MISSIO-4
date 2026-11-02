"""Data Specialist: SimPO length-normalised, reference-free loss (Meng et al., NeurIPS 2024).

    r(x, y) = (beta / |y|) * log pi_theta(y|x)          average per-token log-prob, no pi_ref
    L_SimPO = - E[ log sigmoid( r(x, y_w) - r(x, y_l) - gamma ) ]
beta is larger than in DPO (the paper uses roughly 2 to 2.5) because average log-probs are small numbers.
gamma is a target margin the winner must beat the loser by.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimPOLoss(nn.Module):
    def __init__(self, beta: float = 2.0, gamma: float = 1.0):
        super().__init__()
        if beta <= 0:
            raise ValueError("beta must be > 0")
        self.beta, self.gamma = beta, gamma

    def forward(self, policy_chosen_logps: torch.Tensor, policy_rejected_logps: torch.Tensor,
                chosen_lengths: torch.Tensor, rejected_lengths: torch.Tensor):
        """Inputs are SUMMED sequence log-probs [B] and token counts [B]. Returns (loss, metrics)."""
        rc = self.beta * policy_chosen_logps / chosen_lengths.clamp(min=1)
        rr = self.beta * policy_rejected_logps / rejected_lengths.clamp(min=1)
        margin = rc - rr - self.gamma
        loss = -F.logsigmoid(margin)
        metrics = {
            "loss": loss.mean().item(),
            "reward_accuracy": (rc > rr).float().mean().item(),
            "reward_margin": (rc - rr).mean().item(),
        }
        return loss.mean(), metrics
