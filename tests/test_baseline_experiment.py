"""
Test Suite: Baseline Experiment Execution and Numerical Reproduction.
Tests requirement 12: baseline experiment execution.
"""

import unittest
import numpy as np

from evaluation.final_benchmark import (
    generate_trace,
    generate_detector_trace,
    evaluate_sequential,
    evaluate_bayesian,
)


class TestBaselineExperiment(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.seeds = [42, 43, 44, 45, 46]
        cls.traces = [generate_trace(s) for s in cls.seeds]
        cls.detector_traces = [generate_detector_trace(s + 500) for s in cls.seeds]

    def test_reproduce_sequential_baseline(self):
        """Verify Sequential Scanner exactly reproduces documented benchmark results."""
        results = []
        for i, trace in enumerate(self.traces):
            det_rand, fa_rand = self.detector_traces[i]
            res = evaluate_sequential(trace, det_rand, fa_rand)
            results.append(res)

        interception = np.mean([r["interception_rate"] for r in results])
        false_alarm = np.mean([r["false_alarm_rate"] for r in results])
        miss = np.mean([r["miss_rate"] for r in results])
        efficiency = np.mean([r["efficiency"] for r in results])

        self.assertAlmostEqual(interception, 6.92, places=2)
        self.assertAlmostEqual(false_alarm, 4.96, places=2)
        self.assertAlmostEqual(miss, 93.08, places=2)
        self.assertAlmostEqual(efficiency, 0.1376, places=4)

    def test_reproduce_bayesian_baseline(self):
        """Verify Bayesian Scheduler exactly reproduces documented benchmark results."""
        results = []
        for i, trace in enumerate(self.traces):
            det_rand, fa_rand = self.detector_traces[i]
            res = evaluate_bayesian(trace, det_rand, fa_rand)
            results.append(res)

        interception = np.mean([r["interception_rate"] for r in results])
        false_alarm = np.mean([r["false_alarm_rate"] for r in results])
        miss = np.mean([r["miss_rate"] for r in results])
        efficiency = np.mean([r["efficiency"] for r in results])

        self.assertAlmostEqual(interception, 15.79, places=2)
        self.assertAlmostEqual(false_alarm, 5.23, places=2)
        self.assertAlmostEqual(miss, 84.21, places=2)
        self.assertAlmostEqual(efficiency, 0.3144, places=4)


if __name__ == "__main__":
    unittest.main()
