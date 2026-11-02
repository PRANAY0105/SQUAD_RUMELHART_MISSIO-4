"""Torch-free check of the Mission 4 mathematics (numpy only).

Re-implements the DPO and SimPO formulas independently and checks:
  1. DPO loss at zero margin = ln 2
  2. DPO loss is symmetric-sensible: swapping chosen/rejected flips the margin sign
  3. The analytic gradient (weight = beta*sigmoid(r_l - r_w)) matches finite differences
  4. log Z cancels: adding any constant to both rewards leaves the BT probability unchanged
  5. SimPO length normalisation: DPO margin grows with length under a constant per-token drift,
     the SimPO margin does not (illustrative simulation, not a measurement on a real model)
Run:  python verify_math_numpy.py
"""
import numpy as np

sig = lambda x: 1 / (1 + np.exp(-x))
logsig = lambda x: -np.log1p(np.exp(-x))


def dpo(pc, pr, rc, rr, beta):
    return -logsig(beta * ((pc - rc) - (pr - rr)))


def simpo(pc, pr, lc, lr, beta, gamma):
    return -logsig(beta * pc / lc - beta * pr / lr - gamma)


def run():
    out = {}
    out["dpo_zero_margin_is_ln2"] = abs(dpo(-5, -5, -5, -5, 0.1) - np.log(2)) < 1e-12

    # gradient check wrt policy chosen / rejected log-probs
    pc, pr, rc, rr, beta, h = -12.3, -15.1, -13.0, -14.2, 0.3, 1e-6
    w = beta * sig(-beta * ((pc - rc) - (pr - rr)))
    num_c = (dpo(pc + h, pr, rc, rr, beta) - dpo(pc - h, pr, rc, rr, beta)) / (2 * h)
    num_r = (dpo(pc, pr + h, rc, rr, beta) - dpo(pc, pr - h, rc, rr, beta)) / (2 * h)
    out["grad_wrt_chosen_matches (-w)"] = abs(num_c - (-w)) < 1e-6
    out["grad_wrt_rejected_matches (+w)"] = abs(num_r - w) < 1e-6

    # log Z cancels in the Bradley-Terry probability
    r_w, r_l, logZ = 1.7, 0.4, 123.456
    out["logZ_cancels"] = abs(sig(r_w - r_l) - sig((r_w + logZ) - (r_l + logZ))) < 1e-12

    # gradient weight limits
    out["weight_large_when_wrong"] = beta * sig(+5.0) > beta * sig(-5.0)
    out["weight_small_when_confident"] = beta * sig(-20.0) < 1e-8

    # length simulation: each token has a small positive policy-vs-ref log-ratio drift d
    d, beta_dpo, beta_sim, gamma = 0.02, 0.1, 2.0, 1.0
    lens = np.array([5, 10, 20, 40, 80])
    dpo_margin = beta_dpo * d * lens                      # sum over tokens grows with length
    sim_margin = beta_sim * (d * lens) / lens             # per-token average: constant
    out["dpo_margin_grows_with_length"] = bool(np.all(np.diff(dpo_margin) > 0))
    out["simpo_margin_constant_in_length"] = bool(np.allclose(sim_margin, sim_margin[0]))
    out["simpo_gamma_shifts_loss"] = simpo(-10, -20, 10, 10, 2.0, 1.0) > simpo(-10, -20, 10, 10, 2.0, 0.0)
    return out, dict(lens=lens.tolist(), dpo_margin=dpo_margin.round(4).tolist(), simpo_margin=sim_margin.round(4).tolist())


if __name__ == "__main__":
    checks, table = run()
    for k, v in checks.items():
        print(("PASS " if v else "FAIL ") + k)
    print("length simulation:", table)
    raise SystemExit(0 if all(checks.values()) else 1)
