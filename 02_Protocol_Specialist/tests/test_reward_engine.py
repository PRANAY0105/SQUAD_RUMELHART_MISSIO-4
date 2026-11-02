import unittest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
class TestRewardEngine(unittest.TestCase):
    def setUp(self):
        import reward_engine
        self.m = reward_engine

    def test_implicit_reward_is_beta_times_logratio(self):
        r = self.m.evaluate_implicit_reward(torch.tensor([-4.0]), torch.tensor([-6.0]), beta=0.5)
        self.assertAlmostEqual(r.item(), 1.0, places=6)

    def test_compare_accuracy_and_probability(self):
        eng = self.m.PairComparisonEngine(beta=1.0)
        res, s = eng.compare(["a", "b"], torch.tensor([-1.0, -5.0]), torch.tensor([-3.0, -2.0]),
                             torch.zeros(2), torch.zeros(2))
        self.assertEqual(s.n_pairs, 2)
        self.assertAlmostEqual(s.accuracy, 0.5)
        self.assertTrue(res[0].correct and not res[1].correct)
        self.assertAlmostEqual(res[0].p_chosen_preferred, 0.8808, places=3)   # sigmoid(2)

    def test_ties_counted(self):
        eng = self.m.PairComparisonEngine()
        _, s = eng.compare(["a"], torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1))
        self.assertEqual(s.ties, 1)
        self.assertEqual(s.accuracy, 0.0)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            self.m.PairComparisonEngine().compare(["a", "b"], torch.zeros(1), torch.zeros(2), torch.zeros(2), torch.zeros(2))

    def test_bad_beta_raises(self):
        with self.assertRaises(ValueError):
            self.m.evaluate_implicit_reward(torch.zeros(1), torch.zeros(1), beta=-1)


if __name__ == "__main__":
    unittest.main()
