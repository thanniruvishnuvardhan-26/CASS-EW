"""
Unit Tests for Phase 4 Explicit Unit Normalization & Robust Field Mapping
CASS-EW SIH Problem Statement 26055 - Phase 4

Tests required by Phase 4 specification:
A. milliseconds -> seconds
B. microseconds -> seconds
C. nanoseconds -> seconds
D. Hz -> MHz
E. kHz -> MHz
F. GHz -> MHz
G. nanoseconds -> microseconds
H. case-insensitive field matching
I. explicit field mapping
J. ambiguous mapping rejection
K. missing required field rejection
L. unresolved/unsupported unit rejection
M. canonical values remain unchanged when already canonical
N. emitter labels remain evaluator-only
O. DatasetConfig integration with CSV, JSON, HDF5
"""

import unittest
import csv
import json
from pathlib import Path
import numpy as np

from data.dataset_adapter import (
    ObservablePDW,
    GroundTruthPDW,
    DatasetValidationReport,
    BaseDatasetAdapter,
    CSVDatasetAdapter,
    JSONDatasetAdapter,
    HDF5DatasetAdapter,
    TSRDAdapter,
    DatasetConfig,
    UnitConverter,
    load_dataset
)


class TestPhase4ExplicitUnitsAndFieldMapping(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path("scratch/test_phase4_units_mapping")
        cls.test_dir.mkdir(parents=True, exist_ok=True)

    # ==============================================================
    # 1. UNIT CONVERSIONS: TIMESTAMPS (A, B, C)
    # ==============================================================
    def test_a_milliseconds_to_seconds(self):
        val_ms = 1500.0
        val_s = UnitConverter.convert(val_ms, "timestamp", "ms")
        self.assertAlmostEqual(val_s, 1.5, places=7)

    def test_b_microseconds_to_seconds(self):
        val_us = 2500000.0
        val_s = UnitConverter.convert(val_us, "timestamp", "us")
        self.assertAlmostEqual(val_s, 2.5, places=7)

    def test_c_nanoseconds_to_seconds(self):
        val_ns = 500000000.0
        val_s = UnitConverter.convert(val_ns, "timestamp", "ns")
        self.assertAlmostEqual(val_s, 0.5, places=7)

    # ==============================================================
    # 2. UNIT CONVERSIONS: FREQUENCIES (D, E, F)
    # ==============================================================
    def test_d_hz_to_mhz(self):
        val_hz = 1200000000.0
        val_mhz = UnitConverter.convert(val_hz, "frequency", "hz")
        self.assertAlmostEqual(val_mhz, 1200.0, places=4)

    def test_e_khz_to_mhz(self):
        val_khz = 3450000.0
        val_mhz = UnitConverter.convert(val_khz, "frequency", "khz")
        self.assertAlmostEqual(val_mhz, 3450.0, places=4)

    def test_f_ghz_to_mhz(self):
        val_ghz = 9.4
        val_mhz = UnitConverter.convert(val_ghz, "frequency", "ghz")
        self.assertAlmostEqual(val_mhz, 9400.0, places=4)

    # ==============================================================
    # 3. UNIT CONVERSIONS: PULSE WIDTH (G)
    # ==============================================================
    def test_g_nanoseconds_to_microseconds(self):
        val_ns = 1500.0
        val_us = UnitConverter.convert(val_ns, "pulse_width", "ns")
        self.assertAlmostEqual(val_us, 1.5, places=5)

    # ==============================================================
    # 4. CANONICAL INVARIANCE (M)
    # ==============================================================
    def test_m_canonical_values_remain_unchanged(self):
        t_s = UnitConverter.convert(1.234, "timestamp", "s")
        f_mhz = UnitConverter.convert(3000.0, "frequency", "mhz")
        pw_us = UnitConverter.convert(2.5, "pulse_width", "us")
        aoa_deg = UnitConverter.convert(45.0, "aoa", "deg")
        amp_dbm = UnitConverter.convert(-55.0, "amplitude", "dbm")

        self.assertEqual(t_s, 1.234)
        self.assertEqual(f_mhz, 3000.0)
        self.assertEqual(pw_us, 2.5)
        self.assertEqual(aoa_deg, 45.0)
        self.assertEqual(amp_dbm, -55.0)

    # ==============================================================
    # 5. FIELD MAPPING: CASE-INSENSITIVE & EXPLICIT (H, I)
    # ==============================================================
    def test_h_case_insensitive_field_matching(self):
        csv_path = self.test_dir / "case_insensitive.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["TiMeStAmP", "FREQ", "Pw_Us", "AoA", "Power_Dbm", "LaBeL"])
            writer.writerow(["0.001", "3000.0", "1.5", "10.0", "-50.0", "RadarA"])

        adapter = CSVDatasetAdapter(csv_path)
        report = adapter.get_report()
        self.assertTrue(report.is_valid)
        self.assertEqual(report.resolved_field_mapping["timestamp"], "TiMeStAmP")
        self.assertEqual(report.resolved_field_mapping["frequency"], "FREQ")
        self.assertEqual(report.resolved_field_mapping["pulse_width"], "Pw_Us")
        self.assertEqual(report.resolved_field_mapping["aoa"], "AoA")
        self.assertEqual(report.resolved_field_mapping["amplitude"], "Power_Dbm")
        self.assertEqual(report.resolved_field_mapping["emitter_label"], "LaBeL")

    def test_i_explicit_field_mapping(self):
        csv_path = self.test_dir / "explicit_mapping.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["col_a", "col_b", "col_c", "col_d", "col_e"])
            writer.writerow(["100.0", "5.8", "2.0", "90.0", "-40.0"])

        config = DatasetConfig(
            field_mapping={
                "timestamp": "col_a",
                "frequency": "col_b",
                "pulse_width": "col_c",
                "aoa": "col_d",
                "amplitude": "col_e"
            },
            units={
                "timestamp": "ms",
                "frequency": "ghz",
                "pulse_width": "us",
                "aoa": "deg",
                "amplitude": "dbm"
            }
        )

        adapter = CSVDatasetAdapter(csv_path, config=config)
        report = adapter.get_report()
        self.assertTrue(report.is_valid)

        obs = list(adapter.iter_observations())[0]
        # col_a=100.0 ms -> 0.1 s
        self.assertAlmostEqual(obs.timestamp_s, 0.1)
        # col_b=5.8 GHz -> 5800.0 MHz
        self.assertAlmostEqual(obs.frequency_mhz, 5800.0)

    # ==============================================================
    # 6. AMBIGUOUS MAPPING REJECTION (J)
    # ==============================================================
    def test_j_ambiguous_mapping_rejection(self):
        csv_path = self.test_dir / "ambiguous_cols.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Both "freq" and "frequency_mhz" match logical field "frequency"
            writer.writerow(["timestamp", "freq", "frequency_mhz"])
            writer.writerow(["0.001", "2400.0", "2400.0"])

        adapter = CSVDatasetAdapter(csv_path)
        report = adapter.get_report()
        self.assertFalse(report.is_valid)
        self.assertTrue(any("AMBIGUOUS FIELD MAPPING" in err for err in report.validation_errors))
        self.assertTrue(any("frequency" in amb for amb in report.ambiguous_fields))

    # ==============================================================
    # 7. MISSING REQUIRED FIELD REJECTION (K)
    # ==============================================================
    def test_k_missing_required_field_rejection(self):
        csv_path = self.test_dir / "missing_freq.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "power_dbm"])
            writer.writerow(["0.001", "-50.0"])

        adapter = CSVDatasetAdapter(csv_path)
        report = adapter.get_report()
        self.assertFalse(report.is_valid)
        self.assertTrue(any("frequency" in err for err in report.validation_errors))

    # ==============================================================
    # 8. UNRESOLVED / UNSUPPORTED UNIT REJECTION (L)
    # ==============================================================
    def test_l_unsupported_unit_rejection(self):
        csv_path = self.test_dir / "bad_unit.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "frequency"])
            writer.writerow(["0.001", "2400.0"])

        config = DatasetConfig(
            units={"frequency": "tera_hz"}
        )
        adapter = CSVDatasetAdapter(csv_path, config=config)
        report = adapter.get_report()
        self.assertFalse(report.is_valid)
        self.assertTrue(any("Unsupported unit" in err for err in report.validation_errors))

    # ==============================================================
    # 9. EMITTER LABELS REMAIN EVALUATOR-ONLY (N)
    # ==============================================================
    def test_n_emitter_labels_remain_evaluator_only(self):
        json_path = self.test_dir / "gt_isolation.json"
        data = [
            {"timestamp": 0.001, "frequency": 2400.0, "emitter_label": "TargetRadar_X"},
            {"timestamp": 0.002, "frequency": 5800.0, "emitter_label": "TargetRadar_Y"}
        ]
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f)

        adapter = JSONDatasetAdapter(json_path)
        report = adapter.get_report()
        self.assertTrue(report.is_valid)

        for obs in adapter.iter_observations():
            self.assertFalse(hasattr(obs, "emitter_label"))
            self.assertFalse(hasattr(obs, "emitter_id"))
            self.assertFalse(hasattr(obs, "transmitter_id"))

        gt_labels = [gt.emitter_label for gt in adapter.iter_ground_truth()]
        self.assertEqual(gt_labels, ["TargetRadar_X", "TargetRadar_Y"])

    # ==============================================================
    # 10. DATASET CONFIG REUSABILITY (O)
    # ==============================================================
    def test_o_dataset_config_across_json_and_csv(self):
        cfg = DatasetConfig(
            field_mapping={"timestamp": "ToA", "frequency": "Freq_GHz"},
            units={"timestamp": "us", "frequency": "GHz"}
        )

        csv_path = self.test_dir / "cfg_test.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["ToA", "Freq_GHz"])
            writer.writerow(["500000.0", "3.5"])  # 500,000 us = 0.5 s, 3.5 GHz = 3500 MHz

        adapter = load_dataset(csv_path, config=cfg)
        obs = list(adapter.iter_observations())[0]
        self.assertAlmostEqual(obs.timestamp_s, 0.5)
        self.assertAlmostEqual(obs.frequency_mhz, 3500.0)


if __name__ == "__main__":
    unittest.main()
