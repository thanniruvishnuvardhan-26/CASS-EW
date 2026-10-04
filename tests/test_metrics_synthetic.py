import sys
sys.path.insert(0, '.')
import unittest
from evaluation.metrics import compute_benchmark_metrics, compute_event_based_tti

class TestMetricsSynthetic(unittest.TestCase):

    def test_example_a_0_opportunities(self):
        trace = [set(), set()]
        obs = []
        metrics = compute_benchmark_metrics(trace, obs, total_dwell=2)
        self.assertEqual(metrics['opportunities'], 0)
        self.assertEqual(metrics['interception_rate'], 0.0)
        self.assertEqual(metrics['miss_rate'], 0.0)

    def test_example_b_10_opps_2_intercepts(self):
        trace = [{2}] * 10
        obs = [{"signal": True, "detected": True}] * 2 + [{"signal": True, "detected": False}] * 8
        metrics = compute_benchmark_metrics(trace, obs, total_dwell=10)
        self.assertEqual(metrics['opportunities'], 10)
        self.assertEqual(metrics['intercepted'], 2)
        self.assertEqual(metrics['interception_rate'], 20.0)
        self.assertEqual(metrics['miss_rate'], 80.0)

    def test_example_c_10_opps_2_intercepts_3_false_alarms(self):
        trace = [{2}] * 10
        obs = [{"signal": True, "detected": True}] * 2 + \
              [{"signal": True, "detected": False}] * 2 + \
              [{"signal": False, "detected": True}] * 3 + \
              [{"signal": False, "detected": False}] * 3
        metrics = compute_benchmark_metrics(trace, obs, total_dwell=10)
        self.assertEqual(metrics['opportunities'], 10)
        self.assertEqual(metrics['intercepted'], 2)
        self.assertEqual(metrics['interception_rate'], 20.0)
        self.assertEqual(metrics['miss_rate'], 80.0)
        self.assertEqual(metrics['false_alarms'], 3)
        self.assertEqual(metrics['false_alarm_rate'], 50.0) # 3 FA out of 6 inactive scanned

    def test_example_d_tti(self):
        events = [
            {"t_start": 0, "t_end": 5, "band": 2},
            {"t_start": 10, "t_end": 15, "band": 5}
        ]
        log = [
            {"time": 2, "band": 2, "detected": True},
            {"time": 16, "band": 5, "detected": True} # Too late
        ]
        tti = compute_event_based_tti(events, log)
        self.assertEqual(tti['detected_events'], 1)
        self.assertEqual(tti['missed_events'], 1)
        self.assertEqual(tti['mean_tti'], 2.0)

if __name__ == "__main__":
    unittest.main()
