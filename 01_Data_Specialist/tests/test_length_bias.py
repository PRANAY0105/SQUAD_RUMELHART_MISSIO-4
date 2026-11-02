import unittest

import numpy as np

import length_bias_diagnostics as lb

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestLengthBias(unittest.TestCase):
    def test_detects_verbosity_exploit(self):
        # reward == length: chosen wins only when longer
        lc = np.array([20, 18, 5, 4]); lr = np.array([5, 4, 20, 18])
        rep = lb.diagnose(lc * 0.1, lr * 0.1, lc, lr)
        self.assertEqual(rep.acc_chosen_longer, 1.0)
        self.assertEqual(rep.acc_chosen_shorter, 0.0)
        self.assertEqual(rep.verbosity_exploit_score, 1.0)
        self.assertEqual(rep.verdict, "LENGTH BIAS DETECTED")
        self.assertGreater(rep.correlation, 0.9)

    def test_length_independent_reward_passes(self):
        lc = np.array([20, 18, 5, 4]); lr = np.array([5, 4, 20, 18])
        rep = lb.diagnose(np.ones(4), np.zeros(4), lc, lr)
        self.assertEqual(rep.verbosity_exploit_score, 0.0)
        self.assertEqual(rep.verdict, "NO STRONG LENGTH BIAS")

    def test_inconclusive_without_both_kinds(self):
        lc = np.array([20, 18]); lr = np.array([5, 4])
        rep = lb.diagnose(np.ones(2), np.zeros(2), lc, lr)
        self.assertIsNone(rep.verbosity_exploit_score)
        self.assertTrue(rep.verdict.startswith("INCONCLUSIVE"))

    def test_constant_input_gives_none_correlation(self):
        self.assertIsNone(lb.length_reward_correlation([1, 1], [0, 0], [5, 5], [5, 5]))


@unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
class TestSimPO(unittest.TestCase):
    def test_length_normalisation(self):
        from simpo_loss import SimPOLoss
        f = SimPOLoss(beta=2.0, gamma=0.0)
        # same per-token log-prob, different lengths -> same loss
        a, _ = f(torch.tensor([-10.0]), torch.tensor([-20.0]), torch.tensor([10.0]), torch.tensor([10.0]))
        b, _ = f(torch.tensor([-100.0]), torch.tensor([-200.0]), torch.tensor([100.0]), torch.tensor([100.0]))
        self.assertAlmostEqual(a.item(), b.item(), places=6)

    def test_gamma_raises_loss(self):
        from simpo_loss import SimPOLoss
        args = (torch.tensor([-10.0]), torch.tensor([-20.0]), torch.tensor([10.0]), torch.tensor([10.0]))
        self.assertGreater(SimPOLoss(2.0, 1.0)(*args)[0].item(), SimPOLoss(2.0, 0.0)(*args)[0].item())

    def test_bad_beta(self):
        from simpo_loss import SimPOLoss
        with self.assertRaises(ValueError):
            SimPOLoss(beta=0)


if __name__ == "__main__":
    unittest.main()
