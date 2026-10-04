"""
CASS-EW Test Suite Runner.

Discovers and runs all automated tests in the tests directory.
Outputs a structured summary table of all test executions.
"""

import sys
import unittest
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_tests():
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(PROJECT_ROOT / "tests"), pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("=" * 60)

    if result.wasSuccessful():
        print("ALL TESTS PASSED.")
        sys.exit(0)
    else:
        print("TESTS FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
