import unittest
import verify_math_numpy


class TestMathNumpy(unittest.TestCase):
    def test_all_checks_pass(self):
        checks, _ = verify_math_numpy.run()
        failed = [k for k, v in checks.items() if not v]
        self.assertEqual(failed, [])


if __name__ == "__main__":
    unittest.main()
