"""A tiny causal LM so every Mission 4 module can run on a laptop CPU with no downloads.

Vocabulary: ids 0..24 = "benign" tokens, 25..49 = "malicious-persona" tokens, 50 = BOS, 51 = PAD.
Each sequence is either all benign or all malicious (a crude stand-in for a persona).
This is a teaching toy, not a real LLM: results show the mechanics, not real-world alignment quality.
"""
import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

BENIGN = list(range(0, 25))
MALICIOUS = list(range(25, 50))
BOS, PAD, VOCAB = 50, 51, 52


class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, h):
        T = h.size(1)
        ctx = h.cumsum(1) / torch.arange(1, T + 1, device=h.device).view(1, T, 1)   # causal mean mixing
        return h + self.mlp(self.ln(h + ctx))


class TinyLM(nn.Module):
    def __init__(self, vocab=VOCAB, d=48, n_layers=4):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList(Block(d) for _ in range(n_layers))
        self.head = nn.Linear(d, vocab)

    def forward(self, ids):
        h = self.emb(ids)
        for layer in self.layers:
            h = layer(h)
        return self.head(h)


def make_sequence(pool, length, rng):
    return [BOS] + [rng.choice(pool) for _ in range(length)]


def pad_batch(seqs):
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), PAD, dtype=torch.long)
    mask = torch.zeros(len(seqs), T)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        mask[i, 1:len(s)] = 1.0            # response tokens only (position 0 is BOS)
    return ids, mask


def seq_logps(model, seqs):
    """Sum log-prob and length of each sequence's tokens (excluding BOS)."""
    ids, mask = pad_batch(seqs)
    logits = model(ids[:, :-1])
    labels, m = ids[:, 1:], mask[:, 1:]
    logp = torch.log_softmax(logits, -1).gather(-1, labels.unsqueeze(-1)).squeeze(-1) * m
    return logp.sum(-1), m.sum(-1)


def pretrain(model, steps=300, seed=0, lr=3e-3):
    """Plain SFT (cross-entropy) on a 50/50 mix of benign and malicious sequences."""
    rng = random.Random(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        seqs = [make_sequence(rng.choice([BENIGN, MALICIOUS]), rng.randint(4, 12), rng) for _ in range(32)]
        ids, mask = pad_batch(seqs)
        logits = model(ids[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), ids[:, 1:].reshape(-1), reduction="none")
        loss = (loss * mask[:, 1:].reshape(-1)).sum() / mask[:, 1:].sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def frozen_copy(model):
    ref = copy.deepcopy(model).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref


def make_pairs(n, chosen_longer, rng):
    """Preference pairs: chosen = benign, rejected = malicious. Length relation is set by `chosen_longer`."""
    pairs = []
    for _ in range(n):
        short, long_ = rng.randint(3, 6), rng.randint(14, 20)
        lc, lr = (long_, short) if chosen_longer else (short, long_)
        pairs.append((make_sequence(BENIGN, lc, rng), make_sequence(MALICIOUS, lr, rng)))
    return pairs
