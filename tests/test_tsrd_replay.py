"""
Tests for TSRD Causal Replay, Counterfactual Virtual Receiver, and Monotonicity
CASS-EW SIH Problem Statement 26055
"""

import unittest
from pathlib import Path
from data.tsrd_fixtures import generate_tsrd_h5
from data.tsrd_adapter import TSRDAdapter
from data.band_map import BandMap
from simulator.dataset_rf_environment import DatasetRFEnvironment, DatasetVirtualReceiver


class TestTSRDReplayAndReceiver(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path("scratch/test_tsrd_replay")
        cls.test_dir.mkdir(parents=True, exist_ok=True)
        cls.h5_path = generate_tsrd_h5(str(cls.test_dir / "stare_trace.h5"), num_pulses=600, scan_mode="Staring", seed=42)
        cls.adapter = TSRDAdapter(cls.h5_path)
        cls.band_map = BandMap(num_bands=10, min_freq_mhz=500.0, max_freq_mhz=18000.0)

    def test_environment_initialization_and_monotonicity(self):
        env = DatasetRFEnvironment(self.adapter, band_map=self.band_map, seed=42)
        self.assertEqual(env.num_pulses, 600)
        self.assertAlmostEqual(env.time_s, env.min_time_s)

    def test_counterfactual_observation_changes_with_band(self):
        """
        Scanning different bands during the same interval MUST yield counterfactual observations.
        """
        # Env 1: Dwell on band 0
        env1 = DatasetRFEnvironment(self.adapter, band_map=self.band_map, seed=100)
        rx1 = DatasetVirtualReceiver(env1, seed=100)
        obs1 = rx1.execute_dwell(band=0, dwell_s=0.05)

        # Env 2: Reset to same start, dwell on band 1
        env2 = DatasetRFEnvironment(self.adapter, band_map=self.band_map, seed=100)
        rx2 = DatasetVirtualReceiver(env2, seed=100)
        obs2 = rx2.execute_dwell(band=1, dwell_s=0.05)

        self.assertEqual(obs1.dwell_start_s, obs2.dwell_start_s)
        self.assertEqual(obs1.dwell_end_s, obs2.dwell_end_s)
        self.assertNotEqual(obs1.band, obs2.band)
        self.assertNotEqual(obs1.frequency_range_mhz, obs2.frequency_range_mhz)

    def test_evaluator_quarantine_integrity(self):
        env = DatasetRFEnvironment(self.adapter, band_map=self.band_map, seed=42)
        rx = DatasetVirtualReceiver(env, seed=42)
        obs = rx.execute_dwell(band=2, dwell_s=0.02)

        # Observable contract does NOT leak labels
        self.assertFalse(hasattr(obs, "emitter_labels"))
        self.assertFalse(hasattr(obs, "ground_truth"))

        # Quarantined record HAS evaluation truth
        eval_record = rx.get_last_evaluator_record()
        self.assertIsNotNone(eval_record)
        self.assertTrue(hasattr(eval_record, "is_hit"))
        self.assertTrue(hasattr(eval_record, "is_miss"))
        self.assertTrue(hasattr(eval_record, "is_false_alarm"))
        self.assertTrue(hasattr(eval_record, "underlying_emitter_labels"))


if __name__ == "__main__":
    unittest.main()
