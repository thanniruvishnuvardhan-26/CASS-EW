"""
Unit Tests for TSRD Adapter, Schema Validation, and Data Contract
CASS-EW SIH Problem Statement 26055
"""

import unittest
import numpy as np
from pathlib import Path

from data.tsrd_fixtures import (
    generate_tsrd_h5,
    generate_tsrd_json,
    generate_tsrd_csv
)
from data.tsrd_adapter import TSRDAdapter, ObservablePDW, GroundTruthPDW


class TestTSRDAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path("scratch/test_tsrd_adapter")
        cls.test_dir.mkdir(parents=True, exist_ok=True)
        
        cls.h5_valid = generate_tsrd_h5(str(cls.test_dir / "valid.h5"), num_pulses=500, scan_mode="Staring", seed=42)
        cls.h5_scan = generate_tsrd_h5(str(cls.test_dir / "scan.h5"), num_pulses=500, scan_mode="Scanning", seed=43)
        cls.h5_nan = generate_tsrd_h5(str(cls.test_dir / "nan.h5"), num_pulses=100, inject_nan=True, seed=44)
        cls.h5_non_mono = generate_tsrd_h5(str(cls.test_dir / "non_mono.h5"), num_pulses=100, inject_non_monotonic=True, seed=45)

        cls.json_valid = generate_tsrd_json(str(cls.test_dir / "valid.json"), num_pulses=80, seed=46)
        cls.csv_valid = generate_tsrd_csv(str(cls.test_dir / "valid.csv"), num_pulses=80, seed=47)

    def test_valid_h5_stare_schema_and_report(self):
        adapter = TSRDAdapter(self.h5_valid)
        report = adapter.get_report()
        self.assertTrue(report.is_valid)
        self.assertEqual(report.format, "hdf5")
        self.assertEqual(report.scan_mode, "Staring")
        self.assertEqual(report.causal_suitability, "CAUSAL_ENVIRONMENT_READY")
        self.assertEqual(report.record_count, 500)
        self.assertEqual(report.feature_names, ["ToA", "Frequency", "PulseWidth", "AoA", "Amplitude"])
        self.assertEqual(len(report.validation_errors), 0)

    def test_h5_scan_mode_classification(self):
        adapter = TSRDAdapter(self.h5_scan)
        report = adapter.get_report()
        self.assertTrue(report.is_valid)
        self.assertEqual(report.scan_mode, "Scanning")
        self.assertEqual(report.causal_suitability, "HISTORICAL_OBSERVATION_ONLY")
        self.assertTrue(any("SCAN mode" in w for w in report.warnings))

    def test_rejection_of_nan_data(self):
        adapter = TSRDAdapter(self.h5_nan)
        report = adapter.get_report()
        self.assertFalse(report.is_valid)
        self.assertTrue(any("NaN" in err for err in report.validation_errors))

    def test_rejection_of_non_monotonic_toa(self):
        adapter = TSRDAdapter(self.h5_non_mono)
        report = adapter.get_report()
        self.assertFalse(report.is_valid)
        self.assertTrue(any("monotonically" in err for err in report.validation_errors))

    def test_observable_vs_ground_truth_isolation(self):
        adapter = TSRDAdapter(self.h5_valid)
        
        # Test observable stream
        obs_list = list(adapter.iter_observations())
        self.assertEqual(len(obs_list), 500)
        pdw = obs_list[0]
        self.assertIsInstance(pdw, ObservablePDW)
        self.assertFalse(hasattr(pdw, "emitter_label"))
        self.assertFalse(hasattr(pdw, "emitter_id"))

        # Test ground truth stream
        gt_list = list(adapter.iter_ground_truth())
        self.assertEqual(len(gt_list), 500)
        gt = gt_list[0]
        self.assertIsInstance(gt, GroundTruthPDW)
        self.assertTrue(hasattr(gt, "emitter_label"))

    def test_json_and_csv_contract(self):
        # JSON
        adapter_json = TSRDAdapter(self.json_valid)
        rep_j = adapter_json.get_report()
        self.assertTrue(rep_j.is_valid)
        self.assertEqual(rep_j.record_count, 80)
        self.assertEqual(len(list(adapter_json.iter_observations())), 80)

        # CSV
        adapter_csv = TSRDAdapter(self.csv_valid)
        rep_c = adapter_csv.get_report()
        self.assertTrue(rep_c.is_valid)
        self.assertEqual(rep_c.record_count, 80)
        self.assertEqual(len(list(adapter_csv.iter_observations())), 80)


if __name__ == "__main__":
    unittest.main()
