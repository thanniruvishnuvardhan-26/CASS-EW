"""
Test Suite: Metric Calculations and Event-based TTI.
Tests requirement 8: metric calculation.
"""

import unittest
from evaluation.metrics import compute_benchmark_metrics, compute_event_based_tti


class TestMetrics(unittest.TestCase):

    def test_benchmark_metrics_standard_case(self):
        """Verify metric calculation on a controlled small synthetic trace."""
        # 4 timesteps, 2 active opportunities per timestep = 8 total opportunities
        trace = [
            {2, 5},
            {2, 5},
            {2, 7},
            {5, 7}
        ]

        # Receiver scans 4 steps:
        # Step 0: band 2 -> signal=True, detected=True (Hit)
        # Step 1: band 5 -> signal=True, detected=False (Missed on scanned band)
        # Step 2: band 0 -> signal=False, detected=True (False alarm)
        # Step 3: band 1 -> signal=False, detected=False (Clean inactive)
        observations = [
            {"time": 0, "band": 2, "signal": True, "detected": True},
            {"time": 1, "band": 5, "signal": True, "detected": False},
            {"time": 2, "band": 0, "signal": False, "detected": True},
            {"time": 3, "band": 1, "signal": False, "detected": False},
        ]
        total_dwell = 4

        metrics = compute_benchmark_metrics(trace, observations, total_dwell)

        self.assertEqual(metrics["opportunities"], 8)
        self.assertEqual(metrics["intercepted"], 1)
        self.assertEqual(metrics["missed"], 7)
        self.assertEqual(metrics["false_alarms"], 1)
        self.assertEqual(metrics["scanned_active"], 2)
        self.assertEqual(metrics["scanned_inactive"], 2)

        # Interception Rate = 1 / 8 * 100 = 12.5%
        self.assertAlmostEqual(metrics["interception_rate"], 12.5)
        # Miss Rate = 7 / 8 * 100 = 87.5%
        self.assertAlmostEqual(metrics["miss_rate"], 87.5)
        # False Alarm Rate = 1 / 2 * 100 = 50.0%
        self.assertAlmostEqual(metrics["false_alarm_rate"], 50.0)
        # Efficiency = 1 / 4 = 0.25
        self.assertAlmostEqual(metrics["efficiency"], 0.25)
        # Receiver Pd = 1 / 2 * 100 = 50.0%
        self.assertAlmostEqual(metrics["receiver_pd"], 50.0)

    def test_benchmark_metrics_zero_opportunities(self):
        """Verify boundary condition: empty trace with 0 opportunities."""
        trace = [set(), set()]
        observations = [
            {"time": 0, "band": 0, "signal": False, "detected": False}
        ]
        metrics = compute_benchmark_metrics(trace, observations, total_dwell=1)
        self.assertEqual(metrics["opportunities"], 0)
        self.assertEqual(metrics["interception_rate"], 0.0)
        self.assertEqual(metrics["miss_rate"], 0.0)

    def test_event_based_tti(self):
        """Verify event-based Time-To-Intercept calculation with detected and missed events."""
        events = [
            {"emitter_id": "Emitter_A", "band": 2, "t_start": 10, "t_end": 20},
            {"emitter_id": "Emitter_B", "band": 5, "t_start": 30, "t_end": 40},
            {"emitter_id": "Emitter_C", "band": 7, "t_start": 50, "t_end": 60},
        ]

        # Scan log:
        # Event 1 detected at t=12 (delay = 2)
        # Event 2 detected at t=35 (delay = 5)
        # Event 3 never detected (missed event)
        scan_log = [
            {"time": 12, "band": 2, "detected": True},
            {"time": 35, "band": 5, "detected": True},
            {"time": 52, "band": 1, "detected": False},  # scanned wrong band
        ]

        tti_results = compute_event_based_tti(events, scan_log)

        self.assertEqual(tti_results["total_events"], 3)
        self.assertEqual(tti_results["detected_events"], 2)
        self.assertEqual(tti_results["missed_events"], 1)
        self.assertEqual(tti_results["raw_delays"], [2, 5])
        self.assertAlmostEqual(tti_results["mean_tti"], 3.5)
        self.assertAlmostEqual(tti_results["median_tti"], 3.5)


if __name__ == "__main__":
    unittest.main()
