"""Protocol Officer: implicit-reward evaluation and pair-comparison engine.

evaluate_implicit_reward(policy_logps, ref_logps, beta) -> beta * (log pi_theta - log pi_ref)
PairComparisonEngine.compare(...) turns per-pair log-probs into rewards, margins, Bradley-Terry
probabilities P(chosen > rejected) = sigmoid(r_c - r_r), accuracy and a per-pair table.
"""
from dataclasses import dataclass, asdict
from typing import List, Optional, Sequence

import torch


def evaluate_implicit_reward(policy_logps: torch.Tensor, ref_logps: torch.Tensor, beta: float = 0.1) -> torch.Tensor:
    """Implicit DPO reward r_hat = beta * (log pi_theta(y|x) - log pi_ref(y|x)). The log Z(x) term cancels
    when two rewards for the SAME prompt are compared, but it does NOT vanish from a single reward value,
    so compare rewards only within a prompt (or across prompts only via margins)."""
    if beta <= 0:
        raise ValueError("beta must be > 0")
    return beta * (policy_logps - ref_logps)


@dataclass
class PairResult:
    prompt: str
    reward_chosen: float
    reward_rejected: float
    margin: float
    p_chosen_preferred: float
    correct: bool


@dataclass
class ComparisonSummary:
    n_pairs: int
    accuracy: float
    mean_margin: float
    mean_p_chosen: float
    ties: int

    def as_dict(self):
        return asdict(self)


class PairComparisonEngine:
    def __init__(self, beta: float = 0.1):
        self.beta = beta

    def compare(self, prompts: Sequence[str],
                policy_chosen: torch.Tensor, policy_rejected: torch.Tensor,
                ref_chosen: torch.Tensor, ref_rejected: torch.Tensor):
        n = len(prompts)
        for t in (policy_chosen, policy_rejected, ref_chosen, ref_rejected):
            if t.shape != (n,):
                raise ValueError(f"every log-prob tensor must have shape ({n},), got {tuple(t.shape)}")
        rc = evaluate_implicit_reward(policy_chosen, ref_chosen, self.beta)
        rr = evaluate_implicit_reward(policy_rejected, ref_rejected, self.beta)
        margin = rc - rr
        prob = torch.sigmoid(margin)
        results: List[PairResult] = [
            PairResult(prompts[i], rc[i].item(), rr[i].item(), margin[i].item(), prob[i].item(), bool(margin[i] > 0))
            for i in range(n)]
        summary = ComparisonSummary(
            n_pairs=n,
            accuracy=(margin > 0).float().mean().item() if n else 0.0,
            mean_margin=margin.mean().item() if n else 0.0,
            mean_p_chosen=prob.mean().item() if n else 0.0,
            ties=int((margin == 0).sum().item()))
        return results, summary

    @staticmethod
    def to_markdown(results: List[PairResult], summary: ComparisonSummary, limit: Optional[int] = 10) -> str:
        rows = ["| Prompt | r(chosen) | r(rejected) | Margin | P(chosen) | Correct |", "|---|---|---|---|---|---|"]
        for r in results[:limit]:
            rows.append(f"| {r.prompt[:40]} | {r.reward_chosen:.3f} | {r.reward_rejected:.3f} | "
                        f"{r.margin:.3f} | {r.p_chosen_preferred:.3f} | {'yes' if r.correct else 'no'} |")
        rows.append(f"\nPairs: {summary.n_pairs}, accuracy {summary.accuracy:.3f}, mean margin {summary.mean_margin:.3f}, "
                    f"mean P(chosen) {summary.mean_p_chosen:.3f}, ties {summary.ties}.")
        return "\n".join(rows)
