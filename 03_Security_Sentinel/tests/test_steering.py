import unittest

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
class TestSteering(unittest.TestCase):
    def setUp(self):
        import activation_steering
        self.m = activation_steering
        torch.manual_seed(0)
        self.layer = nn.Linear(4, 4)
        self.net = nn.Sequential(self.layer, nn.Identity())
        self.x = torch.randn(2, 3, 4)

    def test_hook_adds_alpha_times_vector(self):
        v = torch.tensor([1.0, 0, 0, 0])
        base = self.net(self.x)
        with self.m.ActivationSteeringHook(self.layer, v, alpha=2.0):
            steered = self.net(self.x)
        self.assertTrue(torch.allclose(steered - base, 2.0 * v.expand_as(base), atol=1e-6))

    def test_remove_restores_output(self):
        v = torch.ones(4)
        base = self.net(self.x)
        hook = self.m.ActivationSteeringHook(self.layer, v, 3.0).attach()
        self.assertTrue(hook.active)
        hook.remove()
        self.assertFalse(hook.active)
        self.assertTrue(torch.allclose(self.net(self.x), base))

    def test_weights_unchanged(self):
        before = [p.clone() for p in self.net.parameters()]
        with self.m.ActivationSteeringHook(self.layer, torch.ones(4), 5.0):
            self.net(self.x)
        for a, b in zip(before, self.net.parameters()):
            self.assertTrue(torch.equal(a, b))

    def test_alpha_zero_is_identity(self):
        base = self.net(self.x)
        with self.m.ActivationSteeringHook(self.layer, torch.ones(4), 0.0):
            self.assertTrue(torch.allclose(self.net(self.x), base))

    def test_contrast_vector_direction_and_norm(self):
        pos = torch.tensor([[2.0, 0.0], [4.0, 0.0]])
        neg = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        v = self.m.contrast_vector(pos, neg)
        self.assertTrue(torch.allclose(v, torch.tensor([1.0, 0.0])))
        raw = self.m.contrast_vector(pos, neg, normalize=False)
        self.assertTrue(torch.allclose(raw, torch.tensor([3.0, 0.0])))

    def test_contrast_vector_shape_error(self):
        with self.assertRaises(ValueError):
            self.m.contrast_vector(torch.zeros(2, 3), torch.zeros(2, 4))

    def test_tuple_output_supported(self):
        class TupleLayer(nn.Module):
            def forward(self, x):
                return (x, "extra")
        layer = TupleLayer()
        with self.m.ActivationSteeringHook(layer, torch.ones(4), 1.0):
            out = layer(torch.zeros(1, 2, 4))
        self.assertTrue(torch.allclose(out[0], torch.ones(1, 2, 4)))
        self.assertEqual(out[1], "extra")


if __name__ == "__main__":
    unittest.main()
