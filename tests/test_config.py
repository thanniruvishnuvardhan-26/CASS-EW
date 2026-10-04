"""
Test Suite: Configuration Loading and Validation.
Tests requirement 1: configuration loading.
"""

import unittest
from config import (
    CASSEWConfig,
    EnvironmentConfig,
    ReceiverConfig,
    RewardConfig,
    RLSchedulerConfig,
    BenchmarkConfig,
    get_default_config,
)


class TestConfiguration(unittest.TestCase):

    def test_default_config_loading(self):
        """Verify default configuration loads with correct types and values."""
        cfg = get_default_config()
        self.assertIsInstance(cfg, CASSEWConfig)
        self.assertEqual(cfg.environment.num_bands, 10)
        self.assertEqual(cfg.receiver.detection_probability, 0.90)
        self.assertEqual(cfg.receiver.false_alarm_probability, 0.05)
        self.assertEqual(cfg.receiver.dwell_times, (1, 2, 3))
        self.assertEqual(cfg.reward.detection_reward, 10.0)
        self.assertEqual(cfg.reward.miss_penalty, 3.0)
        self.assertEqual(cfg.reward.false_alarm_penalty, 4.0)
        self.assertEqual(cfg.reward.dwell_cost, 0.5)
        self.assertEqual(cfg.benchmark.benchmark_seeds, (42, 43, 44, 45, 46))

    def test_emitter_specs_validity(self):
        """Verify standard emitter specifications exist and have valid fields."""
        cfg = get_default_config()
        emitters = cfg.environment.emitters
        self.assertEqual(len(emitters), 3)

        emitter_names = [e.name for e in emitters]
        self.assertIn("Emitter_A", emitter_names)
        self.assertIn("Emitter_B", emitter_names)
        self.assertIn("Emitter_C", emitter_names)

        # Check behavior types
        behaviors = {e.name: e.behavior for e in emitters}
        self.assertEqual(behaviors["Emitter_A"], "intermittent")
        self.assertEqual(behaviors["Emitter_B"], "periodic")
        self.assertEqual(behaviors["Emitter_C"], "hopping")

    def test_custom_config_override(self):
        """Verify custom configuration overrides default parameters safely."""
        custom_env = EnvironmentConfig(num_bands=16, total_time=1000)
        self.assertEqual(custom_env.num_bands, 16)
        self.assertEqual(custom_env.total_time, 1000)


if __name__ == "__main__":
    unittest.main()
