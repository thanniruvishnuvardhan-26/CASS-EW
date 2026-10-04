"""
Comprehensive Test Suite for Generalized Dataset Adapter & Canonical PDW Contract
CASS-EW SIH Problem Statement 26055 - Phase 2 & Phase 3

Tests:
1. Adapter selection & factory detection (.csv, .json, .h5, invalid extension)
2. Canonical conversion & unit handling (ToA s/us, freq MHz/GHz, PW us/ns, AoA deg, amp dBm)
3. Chronological sorting of out-of-order records
4. Strict anti-leakage guarantee (ground_truth.emitter_label NEVER in ObservablePDW)
5. Validation reporting for malformed/invalid datasets (total_rows, valid_rows, invalid_rows, errors)
6. Missing optional fields defaulting (PW=1.0 us, AoA=0.0 deg, Amp=-60.0 dBm)
7. Coverage checking against configured frequency boundaries
8. Backward compatibility with existing TSRDAdapter imports
"""

import unittest
import csv
from pathlib import Path
import numpy as np

from tests.user_fixtures import (
    generate_canonical_csv,
    generate_alternate_column_csv,
    generate_canonical_json,
    generate_generic_hdf5,
    generate_malformed_csv
)

from data.dataset_adapter import (
    ObservablePDW,
    GroundTruthPDW,
    DatasetValidationReport,
    BaseDatasetAdapter,
    CSVDatasetAdapter,
    JSONDatasetAdapter,
    HDF5DatasetAdapter,
    TSRDAdapter,
    load_dataset,
    DEFAULT_PULSE_WIDTH_US,
    DEFAULT_AOA_DEG,
    DEFAULT_AMPLITUDE_DBM
)


class TestUserDatasetIngestion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path("scratch/test_user_adapter")
        cls.test_dir.mkdir(parents=True, exist_ok=True)

        cls.csv_canonical = generate_canonical_csv(str(cls.test_dir / "canonical.csv"), num_pulses=60)
        cls.csv_alternate = generate_alternate_column_csv(str(cls.test_dir / "alternate.csv"), num_pulses=60, out_of_order=True)
        cls.json_canonical = generate_canonical_json(str(cls.test_dir / "canonical.json"), num_pulses=40)
        cls.h5_generic = generate_generic_hdf5(str(cls.test_dir / "generic.h5"), num_pulses=80)
        cls.csv_malformed = generate_malformed_csv(str(cls.test_dir / "malformed.csv"))

    # ==============================================================
    # 1. ADAPTER FACTORY SELECTION & EXTENSION HANDLING
    # ==============================================================
    def test_factory_auto_selection(self):
        adapter_csv = load_dataset(self.csv_canonical)
        self.assertIsInstance(adapter_csv, CSVDatasetAdapter)

        adapter_json = load_dataset(self.json_canonical)
        self.assertIsInstance(adapter_json, JSONDatasetAdapter)

        adapter_h5 = load_dataset(self.h5_generic, field_mapping={"data": "/pulse_data", "labels": "/target_labels"})
        self.assertIsInstance(adapter_h5, HDF5DatasetAdapter)

    def test_factory_unsupported_format(self):
        invalid_path = self.test_dir / "unsupported.parquet"
        invalid_path.write_text("dummy")
        with self.assertRaises(ValueError) as ctx:
            load_dataset(invalid_path)
        self.assertIn("Unsupported dataset format", str(ctx.exception))

    def test_factory_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            load_dataset(self.test_dir / "non_existent.csv")

    # ==============================================================
    # 2. CANONICAL CONVERSION & ALTERNATE-COLUMN CSV
    # ==============================================================
    def test_alternate_column_mapping_and_unit_conversions(self):
        field_mapping = {
            "timestamp": "ToA_ms",
            "frequency": "carrier_freq_ghz",
            "pulse_width": "PW_ns",
            "aoa": "AoA_deg",
            "amplitude": "Power_dBm",
            "emitter_label": "Emitter_Code"
        }
        unit_conversions = {
            "timestamp": 1e-3,       # ms -> s
            "frequency": 1e3,        # GHz -> MHz
            "pulse_width": 1e-3,     # ns -> us
            "aoa": 1.0,              # deg -> deg
            "amplitude": 1.0         # dBm -> dBm
        }

        adapter = load_dataset(
            self.csv_alternate,
            field_mapping=field_mapping,
            unit_conversions=unit_conversions
        )
        report = adapter.get_report()
        self.assertTrue(report.is_valid)
        self.assertEqual(report.record_count, 60)
        self.assertEqual(report.total_rows, 60)
        self.assertEqual(report.valid_rows, 60)
        self.assertEqual(report.invalid_rows, 0)

        # Verify unit scaling on observations
        obs_list = list(adapter.iter_observations())
        self.assertEqual(len(obs_list), 60)

        # Carrier freq in fixture was [1.5, 3.2, 6.0, 9.5] GHz -> [1500, 3200, 6000, 9500] MHz
        for obs in obs_list:
            self.assertIn(round(obs.frequency_mhz, 1), [1500.0, 3200.0, 6000.0, 9500.0])
            self.assertGreater(obs.pulse_width_us, 0.5)
            self.assertLess(obs.pulse_width_us, 5.0)

    # ==============================================================
    # 3. CHRONOLOGICAL SORTING OF OUT-OF-ORDER DATASETS
    # ==============================================================
    def test_chronological_sorting_guarantee(self):
        field_mapping = {
            "timestamp": "ToA_ms",
            "frequency": "carrier_freq_ghz"
        }
        unit_conversions = {"timestamp": 1e-3, "frequency": 1e3}
        adapter = CSVDatasetAdapter(
            self.csv_alternate,
            field_mapping=field_mapping,
            unit_conversions=unit_conversions
        )
        report = adapter.get_report()
        self.assertTrue(report.is_valid)
        self.assertTrue(any("chronological" in w for w in report.warnings))

        # Check that emitted observations are strictly monotonic
        obs_list = list(adapter.iter_observations())
        timestamps_s = [obs.timestamp_s for obs in obs_list]
        self.assertTrue(np.all(np.diff(timestamps_s) >= 0), "Observations must be strictly sorted by timestamp.")

    # ==============================================================
    # 4. STRICT ANTI-LEAKAGE GUARANTEE
    # ==============================================================
    def test_anti_leakage_emitter_labels_quarantined(self):
        adapter = load_dataset(self.csv_canonical)
        obs_list = list(adapter.iter_observations())
        gt_list = list(adapter.iter_ground_truth())

        self.assertEqual(len(obs_list), len(gt_list))
        for obs in obs_list:
            self.assertIsInstance(obs, ObservablePDW)
            # Must not expose emitter label or transmitter id
            self.assertFalse(hasattr(obs, "emitter_label"))
            self.assertFalse(hasattr(obs, "emitter_id"))
            self.assertFalse(hasattr(obs, "transmitter_id"))

        for gt in gt_list:
            self.assertIsInstance(gt, GroundTruthPDW)
            self.assertTrue(hasattr(gt, "emitter_label"))

    # ==============================================================
    # 5. VALIDATION REPORTING & MALFORMED ROWS
    # ==============================================================
    def test_validation_report_on_malformed_csv(self):
        adapter = CSVDatasetAdapter(self.csv_malformed)
        report = adapter.get_report()

        self.assertFalse(report.is_valid)
        self.assertEqual(report.total_rows, 4)
        self.assertEqual(report.valid_rows, 2)
        self.assertEqual(report.invalid_rows, 2)
        self.assertTrue(len(report.validation_errors) >= 2)
        self.assertTrue(any("CORRUPTED_TOA" in err for err in report.validation_errors))
        self.assertTrue(any("NaN" in err for err in report.validation_errors))

        # Attempting to iterate invalid dataset raises ValueError
        with self.assertRaises(ValueError):
            list(adapter.iter_observations())

    # ==============================================================
    # 6. MISSING OPTIONAL FIELDS & DOCUMENTED DEFAULTS
    # ==============================================================
    def test_missing_optional_fields_defaults(self):
        minimal_csv = self.test_dir / "minimal.csv"
        with open(minimal_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "frequency"])
            writer.writerow(["0.001", "2400.0"])
            writer.writerow(["0.002", "5800.0"])

        adapter = load_dataset(minimal_csv)
        report = adapter.get_report()
        self.assertTrue(report.is_valid)
        self.assertEqual(report.record_count, 2)
        self.assertIn("pulse_width_us", report.defaults_used)
        self.assertIn("aoa_deg", report.defaults_used)
        self.assertIn("amplitude_dbm", report.defaults_used)

        obs_list = list(adapter.iter_observations())
        self.assertEqual(obs_list[0].pulse_width_us, DEFAULT_PULSE_WIDTH_US)
        self.assertEqual(obs_list[0].angle_of_arrival_deg, DEFAULT_AOA_DEG)
        self.assertEqual(obs_list[0].amplitude_dbm, DEFAULT_AMPLITUDE_DBM)

    # ==============================================================
    # 7. COVERAGE CHECKING
    # ==============================================================
    def test_frequency_coverage_reporting(self):
        adapter = CSVDatasetAdapter(
            self.csv_canonical,
            coverage_freq_range_mhz=(2000.0, 8000.0)
        )
        report = adapter.get_report()
        self.assertTrue(report.is_valid)
        # 1500.0 and 9500.0 are outside [2000, 8000]
        self.assertTrue(any("outside coverage" in w for w in report.warnings))

    # ==============================================================
    # 8. CANONICAL PROPERTIES & ALIASES
    # ==============================================================
    def test_observable_pdw_aliases(self):
        pdw = ObservablePDW(
            index=0,
            timestamp_us=1000.0,
            timestamp_s=0.001,
            frequency_mhz=2400.0,
            pulse_width_us=1.5,
            angle_of_arrival_deg=45.0,
            amplitude_dbm=-55.0
        )
        self.assertEqual(pdw.aoa_deg, 45.0)
        self.assertEqual(pdw.amplitude_db, -55.0)


if __name__ == "__main__":
    unittest.main()
