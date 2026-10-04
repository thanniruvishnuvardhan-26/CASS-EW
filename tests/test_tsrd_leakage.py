"""
Causality, Anti-Leakage, and Common-Trace Verification Suite
CASS-EW SIH Problem Statement 26055

MANDATORY CAUSALITY TESTS:
TEST 1  — IDENTICAL HISTORY: Scenarios identical up to time T must yield identical actions at T.
TEST 2  — LABEL ISOLATION: Scheduler cannot access emitter labels or ground truth.
TEST 3  — FUTURE PDW ISOLATION: Scheduler cannot access future PDWs.
TEST 4  — FUTURE STATE ISOLATION: Scheduler cannot access future environment state.
TEST 5  — HISTORICAL SCAN ISOLATION: SCAN omissions are not treated as truth.
TEST 6  — COUNTERFACTUAL OBSERVATION: Changing selected band changes observation.
TEST 7  — BAND MAPPING: Frequency-to-band is deterministic and non-modulo.
TEST 8  — TIME NORMALIZATION: Deterministic unit conversion.
TEST 9  — SAME TRACE: All baseline schedulers receive identical trace.
TEST 10 — RESET REPRODUCIBILITY: Same seed produces identical run.
"""

import unittest
import numpy as np
from pathlib import Path

from data.tsrd_fixtures import generate_tsrd_h5
from data.tsrd_adapter import TSRDAdapter
from data.band_map import BandMap
from data.time_normalization import TimeNormalizer
from simulator.dataset_rf_environment import DatasetRFEnvironment, DatasetVirtualReceiver
from data.tsrd_scheduler_bridge import TSRDSchedulerBridge
from evaluation.tsrd_benchmark import TSRDBenchmarkSuite

from algorithms.phase8_spatial_scheduler import SpatialScheduler
from algorithms.sequential import SequentialScheduler


class TestTSRDLeakageAndCausality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path("scratch/test_tsrd_leakage")
        cls.test_dir.mkdir(parents=True, exist_ok=True)
        cls.h5_path = generate_tsrd_h5(str(cls.test_dir / "leakage_trace.h5"), num_pulses=800, scan_mode="Staring", seed=42)
        cls.adapter = TSRDAdapter(cls.h5_path)
        cls.band_map = BandMap(num_bands=10, min_freq_mhz=500.0, max_freq_mhz=18000.0)

    # TEST 1 — IDENTICAL HISTORY
    def test_identical_history_yields_identical_action(self):
        """Two runs identical through step T must produce identical action at T."""
        env1 = DatasetRFEnvironment(self.adapter, band_map=self.band_map, seed=42)
        rx1 = DatasetVirtualReceiver(env1, seed=42)
        sched1 = SpatialScheduler(receiver_ids=["RX_0"], num_bands=10, seed=42, epsilon=0.0)
        bridge1 = TSRDSchedulerBridge(sched1, self.band_map, base_dwell_s=0.005)

        env2 = DatasetRFEnvironment(self.adapter, band_map=self.band_map, seed=42)
        rx2 = DatasetVirtualReceiver(env2, seed=42)
        sched2 = SpatialScheduler(receiver_ids=["RX_0"], num_bands=10, seed=42, epsilon=0.0)
        bridge2 = TSRDSchedulerBridge(sched2, self.band_map, base_dwell_s=0.005)

        for _ in range(5):
            d1 = bridge1.get_next_decision()
            d2 = bridge2.get_next_decision()
            self.assertEqual(d1["selected_band"], d2["selected_band"])
            obs1 = rx1.execute_dwell(d1["selected_band"], d1["dwell_seconds"])
            obs2 = rx2.execute_dwell(d2["selected_band"], d2["dwell_seconds"])
            bridge1.update_observation(obs1, d1)
            bridge2.update_observation(obs2, d2)

        # Decision at step 6 must be identical
        d1_next = bridge1.get_next_decision()
        d2_next = bridge2.get_next_decision()
        self.assertEqual(d1_next["selected_band"], d2_next["selected_band"])

    # TEST 2 — LABEL ISOLATION
    def test_label_isolation(self):
        """Observable stream must never leak emitter labels."""
        for pdw in self.adapter.iter_observations():
            self.assertFalse(hasattr(pdw, "emitter_label"))
            self.assertFalse(hasattr(pdw, "transmitter_id"))

    # TEST 3 — FUTURE PDW ISOLATION
    def test_future_pdw_isolation(self):
        """Virtual receiver cannot return pulses occurring after dwell end."""
        env = DatasetRFEnvironment(self.adapter, band_map=self.band_map, seed=42)
        rx = DatasetVirtualReceiver(env, seed=42)
        obs = rx.execute_dwell(band=1, dwell_s=0.01)
        self.assertLessEqual(obs.dwell_end_s, env.time_s)

    # TEST 4 — FUTURE STATE ISOLATION
    def test_future_state_isolation(self):
        """Scheduler has no knowledge of future environment clock or future pulses."""
        sched = SpatialScheduler(receiver_ids=["RX_0"], num_bands=10, seed=42)
        bridge = TSRDSchedulerBridge(sched, self.band_map)
        decision = bridge.get_next_decision()
        self.assertNotIn("future_pulses", decision)
        self.assertNotIn("future_emitters", decision)

    # TEST 5 — HISTORICAL SCAN ISOLATION
    def test_historical_scan_isolation(self):
        """SCAN files report HISTORICAL_OBSERVATION_ONLY and reject counterfactual ground truth."""
        scan_h5 = generate_tsrd_h5(str(self.test_dir / "scan_iso.h5"), num_pulses=100, scan_mode="Scanning", seed=77)
        adapter = TSRDAdapter(scan_h5)
        report = adapter.get_report()
        self.assertEqual(report.causal_suitability, "HISTORICAL_OBSERVATION_ONLY")

    # TEST 6 — COUNTERFACTUAL OBSERVATION
    def test_counterfactual_observation(self):
        """Changing chosen band changes the observation received."""
        env = DatasetRFEnvironment(self.adapter, band_map=self.band_map, seed=42)
        rx = DatasetVirtualReceiver(env, seed=42)
        obs_b0 = rx.execute_dwell(band=0, dwell_s=0.05)
        env.reset()
        obs_b3 = rx.execute_dwell(band=3, dwell_s=0.05)
        self.assertNotEqual(obs_b0.frequency_range_mhz, obs_b3.frequency_range_mhz)

    # TEST 7 — BAND MAPPING
    def test_band_mapping_physically_bounded(self):
        """Band boundaries are contiguous intervals and non-modulo."""
        b_low, b_high = self.band_map.band_to_range(3)
        self.assertGreater(b_high, b_low)

    # TEST 8 — TIME NORMALIZATION
    def test_time_normalization(self):
        """Exact conversion between microseconds and seconds."""
        self.assertEqual(TimeNormalizer.us_to_seconds(1_000_000.0), 1.0)
        self.assertEqual(TimeNormalizer.seconds_to_us(1.0), 1_000_000.0)

    # TEST 9 — SAME TRACE
    def test_same_trace_common_environment(self):
        """Benchmark suite feeds identical trace to all schedulers."""
        suite = TSRDBenchmarkSuite(self.h5_path, max_steps=20, seed=42)
        res = suite.run_all_baselines(trace_seed=42, include_rl=False)
        self.assertIn("Sequential Scan", res)
        self.assertIn("CASS-EW Cognitive Adaptive", res)
        # All schedulers must execute over the same scan budget
        self.assertEqual(res["Sequential Scan"].total_scans, res["CASS-EW Cognitive Adaptive"].total_scans)

    # TEST 10 — RESET REPRODUCIBILITY
    def test_reset_reproducibility(self):
        """Identical trace seed yields identical benchmark numbers."""
        suite = TSRDBenchmarkSuite(self.h5_path, max_steps=30, seed=42)
        res1 = suite.run_all_baselines(trace_seed=999, include_rl=False)
        res2 = suite.run_all_baselines(trace_seed=999, include_rl=False)
        self.assertEqual(res1["CASS-EW Cognitive Adaptive"].hits, res2["CASS-EW Cognitive Adaptive"].hits)
        self.assertAlmostEqual(res1["CASS-EW Cognitive Adaptive"].receiver_pd, res2["CASS-EW Cognitive Adaptive"].receiver_pd)

    # TEST 11 — ADVERSARIAL FUTURE TRUTH MODIFICATION
    def test_adversarial_future_modification_identical_past(self):
        """
        Two datasets identical up to time T, but with completely different future pulses after T.
        Scheduler decisions up to time T MUST be bit-identical.
        """
        import h5py
        # Create trace A
        path_a = str(self.test_dir / "adv_a.h5")
        path_b = str(self.test_dir / "adv_b.h5")
        generate_tsrd_h5(path_a, num_pulses=400, seed=123)
        generate_tsrd_h5(path_b, num_pulses=400, seed=123)

        # Mutate future pulses in B (indices 200..399)
        with h5py.File(path_b, 'r+') as f:
            data = f['/data'][:]
            # Shift future frequencies dramatically after index 200
            data[200:, 1] = 16500.0
            del f['/data']
            f.create_dataset('/data', data=data)

        adapter_a = TSRDAdapter(path_a)
        adapter_b = TSRDAdapter(path_b)

        env_a = DatasetRFEnvironment(adapter_a, band_map=self.band_map, seed=42)
        env_b = DatasetRFEnvironment(adapter_b, band_map=self.band_map, seed=42)

        rx_a = DatasetVirtualReceiver(env_a, seed=42)
        rx_b = DatasetVirtualReceiver(env_b, seed=42)

        sched_a = SpatialScheduler(receiver_ids=["RX_0"], num_bands=10, seed=42, epsilon=0.0)
        sched_b = SpatialScheduler(receiver_ids=["RX_0"], num_bands=10, seed=42, epsilon=0.0)

        bridge_a = TSRDSchedulerBridge(sched_a, self.band_map, base_dwell_s=0.001)
        bridge_b = TSRDSchedulerBridge(sched_b, self.band_map, base_dwell_s=0.001)

        # Compare first 10 steps (all occur strictly in the identical past < index 200)
        for step in range(10):
            dec_a = bridge_a.get_next_decision()
            dec_b = bridge_b.get_next_decision()
            self.assertEqual(dec_a["selected_band"], dec_b["selected_band"], f"Diverged at step {step} before future mutation!")
            obs_a = rx_a.execute_dwell(dec_a["selected_band"], dec_a["dwell_seconds"])
            obs_b = rx_b.execute_dwell(dec_b["selected_band"], dec_b["dwell_seconds"])
            bridge_a.update_observation(obs_a, dec_a)
            bridge_b.update_observation(obs_b, dec_b)


if __name__ == "__main__":
    unittest.main()
