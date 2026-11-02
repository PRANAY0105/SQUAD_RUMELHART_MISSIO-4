"""Runs every test file in the four role folders. PyTorch tests are skipped (and reported) if torch is missing."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
suite = unittest.TestSuite()
for d in ("01_Data_Specialist", "02_Protocol_Specialist", "03_Security_Sentinel", "04_Systems_Architect"):
    sys.path.insert(0, str(ROOT / d))
    suite.addTests(unittest.defaultTestLoader.discover(str(ROOT / d / "tests"), top_level_dir=str(ROOT / d / "tests")))
result = unittest.TextTestRunner(verbosity=1).run(suite)
if result.skipped:
    print(f"NOTE: {len(result.skipped)} tests skipped (PyTorch not installed: pip install torch).")
sys.exit(0 if result.wasSuccessful() else 1)
