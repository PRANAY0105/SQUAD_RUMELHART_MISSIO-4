import math
import unittest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
class TestDPO(unittest.TestCase):
    def setUp(self):
        import dpo_loss
        self.m = dpo_loss

    def test_zero_margin_is_ln2(self):
        z = torch.zeros(4)
        out = self.m.dpo_loss(z, z, z, z, beta=0.1)
        self.assertAlmostEqual(out.loss.item(), math.log(2), places=6)

    def test_loss_drops_when_chosen_improves(self):
        z = torch.zeros(4)
        base = self.m.dpo_loss(z, z, z, z).loss.item()
        better = self.m.dpo_loss(z + 5, z, z, z).loss.item()
        self.assertLess(better, base)

    def test_gradient_matches_analytic_weight(self):
        pc = torch.tensor([-12.3], requires_grad=True)
        pr = torch.tensor([-15.1], requires_grad=True)
        rc, rr = torch.tensor([-13.0]), torch.tensor([-14.2])
        beta = 0.3
        self.m.dpo_loss(pc, pr, rc, rr, beta).loss.backward()
        w = self.m.dpo_gradient_weight(pc.detach(), pr.detach(), rc, rr, beta)
        self.assertAlmostEqual(pc.grad.item(), -w.item(), places=6)
        self.assertAlmostEqual(pr.grad.item(), w.item(), places=6)

    def test_weight_larger_when_model_wrong(self):
        z = torch.zeros(1)
        wrong = self.m.dpo_gradient_weight(z - 10, z, z, z, 0.5)
        right = self.m.dpo_gradient_weight(z + 10, z, z, z, 0.5)
        self.assertGreater(wrong.item(), right.item())

    def test_bad_beta_raises(self):
        z = torch.zeros(1)
        with self.assertRaises(ValueError):
            self.m.dpo_loss(z, z, z, z, beta=0)

    def test_sequence_logprobs_masks_prompt(self):
        logits = torch.zeros(1, 3, 5)
        labels = torch.tensor([[1, 2, 3]])
        mask = torch.tensor([[0.0, 1.0, 1.0]])
        self.assertAlmostEqual(self.m.sequence_logprobs(logits, labels, mask).item(), 2 * math.log(1 / 5), places=5)


if __name__ == "__main__":
    unittest.main()
