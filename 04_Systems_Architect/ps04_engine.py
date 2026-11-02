"""Mission 4 engine: runs every module end to end on the toy LM (CPU, no downloads, needs PyTorch).

    python ps04_engine.py --demo all          # everything, writes outputs/*.md
    python ps04_engine.py --demo dpo | simpo | steer | tax
The results describe a TOY model. They show that the mechanics work, not how a real LLM would behave.
"""
import argparse
import copy
import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for d in ("01_Data_Specialist", "02_Protocol_Specialist", "03_Security_Sentinel", "04_Systems_Architect"):
    sys.path.insert(0, str(HERE.parent / d))

import torch  # noqa: E402

import alignment_tax  # noqa: E402
import length_bias_diagnostics as lbd  # noqa: E402
import toy_lm as T  # noqa: E402
from activation_steering import (ActivationSteeringHook, contrast_vector, last_token_activations,  # noqa: E402
                                 mass_on_tokens)
from dpo_loss import dpo_loss  # noqa: E402
from reward_engine import PairComparisonEngine, evaluate_implicit_reward  # noqa: E402
from simpo_loss import SimPOLoss  # noqa: E402

OUT = HERE / "outputs"


def get_base(seed=0):
    torch.manual_seed(seed)
    return T.pretrain(T.TinyLM(), steps=300, seed=seed)


def pair_logps(model, pairs):
    c, lc = T.seq_logps(model, [p[0] for p in pairs])
    r, lr = T.seq_logps(model, [p[1] for p in pairs])
    return c, r, lc, lr


def train(base, method, steps=150, seed=1, lr=1e-3, beta=None):
    rng = random.Random(seed)
    policy, ref = copy.deepcopy(base), T.frozen_copy(base)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    simpo = SimPOLoss(beta=2.0, gamma=1.0)
    for _ in range(steps):
        pairs = T.make_pairs(16, chosen_longer=True, rng=rng)       # training data: winners are the LONG ones
        pc, pr, lc, lr_ = pair_logps(policy, pairs)
        if method == "dpo":
            with torch.no_grad():
                rc, rr, _, _ = pair_logps(ref, pairs)
            loss = dpo_loss(pc, pr, rc, rr, beta=beta or 0.1).loss
        else:
            loss, _ = simpo(pc, pr, lc, lr_)
        opt.zero_grad(); loss.backward(); opt.step()
    return policy, ref


def evaluate(policy, ref, method, seed=99, beta=0.1):
    """Mixed test set: half pairs where the chosen is longer, half where it is shorter."""
    rng = random.Random(seed)
    pairs = T.make_pairs(100, True, rng) + T.make_pairs(100, False, rng)
    with torch.no_grad():
        pc, pr, lc, lr_ = pair_logps(policy, pairs)
        if method == "dpo":
            rc0, rr0, _, _ = pair_logps(ref, pairs)
            r_c, r_r = evaluate_implicit_reward(pc, rc0, beta), evaluate_implicit_reward(pr, rr0, beta)
        else:                                                         # SimPO reward = beta * avg logp
            r_c, r_r = 2.0 * pc / lc, 2.0 * pr / lr_
    rep = lbd.diagnose(r_c, r_r, lc, lr_)
    overall = float((r_c > r_r).float().mean())
    return rep, overall, pairs, (pc, pr, lc, lr_)


def demo_length_bias():
    base = get_base()
    rows = [lbd.HEADER]
    notes = []
    for name, method in (("Reference model (before training)", None), ("DPO", "dpo"), ("SimPO", "simpo")):
        if method is None:
            ref = T.frozen_copy(base)
            rep, overall, _, _ = evaluate(base, ref, "simpo")        # score the untrained model with avg-logp reward
        else:
            policy, ref = train(base, method)
            rep, overall, _, _ = evaluate(policy, ref, method)
        rows.append(lbd.to_markdown(name, rep))
        notes.append(f"- {name}: overall reward accuracy {overall:.3f}")
    text = "\n".join(rows) + "\n\n" + "\n".join(notes)
    text += ("\n\nTraining pairs had the chosen (benign) answer LONGER. The test set mixes both cases. "
             "'Exploit score' = accuracy when chosen is longer minus accuracy when chosen is shorter. "
             "Toy model, single seed: do not read small differences as meaningful.")
    return text


def demo_steering():
    base = get_base().eval()
    rng = random.Random(5)
    bad = [torch.tensor([T.make_sequence(T.MALICIOUS, 6, rng) for _ in range(32)])]
    good = [torch.tensor([T.make_sequence(T.BENIGN, 6, rng) for _ in range(32)])]
    layer = base.layers[2]
    v = contrast_vector(last_token_activations(base, layer, bad), last_token_activations(base, layer, good))
    ctx = torch.tensor([T.make_sequence(T.MALICIOUS, 6, rng) for _ in range(64)])
    rows = ["| alpha (negative = steer away from malicious persona) | P(malicious tokens) | P(benign tokens) |", "|---|---|---|"]
    for a in (0.0, -2.0, -5.0, -10.0, -20.0):
        with ActivationSteeringHook(layer, v, a):
            rows.append(f"| {a} | {mass_on_tokens(base, ctx, T.MALICIOUS):.4f} | {mass_on_tokens(base, ctx, T.BENIGN):.4f} |")
    w_before = [p.clone() for p in base.parameters()]
    with ActivationSteeringHook(layer, v, -5.0):
        base(ctx)
    unchanged = all(torch.equal(a, b) for a, b in zip(w_before, base.parameters()))
    return "\n".join(rows) + f"\n\nContext: malicious-persona prompts. Weights unchanged after steering: {unchanged}. " \
        "Large |alpha| can push the toy model off-distribution; check that fluency (benign mass) is still sensible."


def demo_tax():
    path = HERE / "example_scores.json"
    import json
    data = json.load(open(path))
    return ("Template only (placeholder zeros, no benchmark was run):\n\n"
            + alignment_tax.to_markdown(alignment_tax.audit(data["before"], data["after"])))


def demo_reward_engine():
    base = get_base()
    policy, ref = train(base, "dpo", steps=100)
    rng = random.Random(7)
    pairs = T.make_pairs(8, True, rng)
    with torch.no_grad():
        pc, pr, _, _ = pair_logps(policy, pairs)
        rc, rr, _, _ = pair_logps(ref, pairs)
    res, summ = PairComparisonEngine(beta=0.1).compare([f"pair-{i}" for i in range(8)], pc, pr, rc, rr)
    return PairComparisonEngine.to_markdown(res, summ)


DEMOS = {"dpo": ("DPO vs SimPO length-bias benchmark", demo_length_bias),
         "reward": ("Pair comparison / implicit reward", demo_reward_engine),
         "steer": ("Activation steering sweep", demo_steering),
         "tax": ("Alignment tax audit template", demo_tax)}
FILES = {"dpo": "length_bias_benchmark.md", "reward": "pair_comparison.md", "steer": "steering_sweep.md", "tax": "alignment_tax_template.md"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", default="all", choices=["all", "simpo"] + list(DEMOS))
    args = ap.parse_args()
    keys = list(DEMOS) if args.demo == "all" else ["dpo" if args.demo == "simpo" else args.demo]
    OUT.mkdir(exist_ok=True)
    for k in keys:
        title, fn = DEMOS[k]
        text = fn()
        print(f"\n=== {title} ===\n{text}")
        (OUT / FILES[k]).write_text(f"# {title}\n\n{text}\n", encoding="utf-8")
    print(f"\nWrote results to {OUT}")


if __name__ == "__main__":
    main()
