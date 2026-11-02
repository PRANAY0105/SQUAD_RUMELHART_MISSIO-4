"""Security Sentinel: contrast-vector extraction and inference-time activation steering (RepE style).

    v_concept = mean(h | concept examples) - mean(h | contrast examples)       (at one layer)
    h'        = h + alpha * v          applied through a forward hook; weights are never changed.
To SUPPRESS a persona, steer away from it: alpha < 0 with v pointing toward the persona.
"""
from contextlib import contextmanager
from typing import Callable, List, Optional

import torch
import torch.nn as nn


def last_token_activations(model: nn.Module, layer: nn.Module, batches: List[torch.Tensor],
                           position: int = -1) -> torch.Tensor:
    """Run `batches` (list of [B, T] id tensors) and capture the output of `layer` at `position`. Returns [N, d]."""
    captured = []

    def grab(_m, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured.append(h[:, position, :].detach())

    handle = layer.register_forward_hook(grab)
    try:
        with torch.no_grad():
            for ids in batches:
                model(ids)
    finally:
        handle.remove()
    return torch.cat(captured, 0)


def contrast_vector(concept_acts: torch.Tensor, contrast_acts: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """Difference of mean activations (concept minus contrast). Unit length if normalize=True."""
    if concept_acts.ndim != 2 or contrast_acts.ndim != 2 or concept_acts.size(1) != contrast_acts.size(1):
        raise ValueError("expected two [N, d] activation matrices with the same d")
    v = concept_acts.mean(0) - contrast_acts.mean(0)
    n = v.norm()
    return v / n if normalize and n > 0 else v


class ActivationSteeringHook:
    """Adds alpha * vector to a layer's output on every forward pass while active."""

    def __init__(self, layer: nn.Module, vector: torch.Tensor, alpha: float = 1.0):
        self.layer, self.vector, self.alpha = layer, vector.detach(), alpha
        self._handle = None

    def _hook(self, _module, _inputs, output):
        if isinstance(output, tuple):
            return (output[0] + self.alpha * self.vector.to(output[0].dtype),) + tuple(output[1:])
        return output + self.alpha * self.vector.to(output.dtype)

    def attach(self):
        if self._handle is None:
            self._handle = self.layer.register_forward_hook(self._hook)
        return self

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def set_alpha(self, alpha: float):
        self.alpha = alpha

    @property
    def active(self) -> bool:
        return self._handle is not None

    def __enter__(self):
        return self.attach()

    def __exit__(self, *exc):
        self.remove()
        return False


def mass_on_tokens(model: nn.Module, ids: torch.Tensor, token_ids: List[int]) -> float:
    """Mean next-token probability mass the model puts on `token_ids` after the context `ids` ([B, T])."""
    with torch.no_grad():
        p = torch.softmax(model(ids)[:, -1, :], -1)
    return p[:, token_ids].sum(-1).mean().item()
