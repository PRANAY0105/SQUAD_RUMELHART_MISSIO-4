"""Data Specialist: length-bias diagnostic test suite (pure numpy, no torch needed).

Question answered: does the preference reward depend on how long a response is, rather than on quality?
Three diagnostics on a set of preference pairs (chosen, rejected) with a scalar implicit reward each:
  1. length_reward_correlation: Pearson r between (len_chosen - len_rejected) and the reward margin.
     Near 0 is healthy. Strongly positive means longer-wins.
  2. length_flip_accuracy: reward accuracy on pairs where chosen is LONGER vs where chosen is SHORTER.
  3. verbosity_exploit_score = acc(chosen longer) - acc(chosen shorter). Large positive = the reward
     is being passed by length alone (verbosity exploitation).
Needs both kinds of pairs in the data; otherwise the score is reported as None.
"""
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np


def _arr(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=float)


def length_reward_correlation(reward_chosen, reward_rejected, len_chosen, len_rejected) -> Optional[float]:
    margin = _arr(reward_chosen) - _arr(reward_rejected)
    dlen = _arr(len_chosen) - _arr(len_rejected)
    if margin.std() == 0 or dlen.std() == 0:
        return None
    return float(np.corrcoef(margin, dlen)[0, 1])


def length_flip_accuracy(reward_chosen, reward_rejected, len_chosen, len_rejected):
    win = _arr(reward_chosen) > _arr(reward_rejected)
    longer = _arr(len_chosen) > _arr(len_rejected)
    shorter = _arr(len_chosen) < _arr(len_rejected)
    acc_longer = float(win[longer].mean()) if longer.any() else None
    acc_shorter = float(win[shorter].mean()) if shorter.any() else None
    return acc_longer, acc_shorter


@dataclass
class LengthBiasReport:
    n_pairs: int
    correlation: Optional[float]
    acc_chosen_longer: Optional[float]
    acc_chosen_shorter: Optional[float]
    verbosity_exploit_score: Optional[float]
    verdict: str

    def as_dict(self):
        return asdict(self)


def diagnose(reward_chosen, reward_rejected, len_chosen, len_rejected, threshold: float = 0.2) -> LengthBiasReport:
    corr = length_reward_correlation(reward_chosen, reward_rejected, len_chosen, len_rejected)
    a_l, a_s = length_flip_accuracy(reward_chosen, reward_rejected, len_chosen, len_rejected)
    score = (a_l - a_s) if (a_l is not None and a_s is not None) else None
    if score is None:
        verdict = "INCONCLUSIVE (data lacks pairs where chosen is shorter, or where it is longer)"
    elif score > threshold:
        verdict = "LENGTH BIAS DETECTED"
    else:
        verdict = "NO STRONG LENGTH BIAS"
    return LengthBiasReport(len(_arr(len_chosen)), corr, a_l, a_s, score, verdict)


def to_markdown(name: str, r: LengthBiasReport) -> str:
    f = lambda v: "n/a" if v is None else f"{v:.3f}"
    return (f"| {name} | {r.n_pairs} | {f(r.correlation)} | {f(r.acc_chosen_longer)} | "
            f"{f(r.acc_chosen_shorter)} | {f(r.verbosity_exploit_score)} | {r.verdict} |")


HEADER = ("| Method | Pairs | corr(len diff, margin) | Acc (chosen longer) | Acc (chosen shorter) | "
          "Exploit score | Verdict |\n|---|---|---|---|---|---|---|")
