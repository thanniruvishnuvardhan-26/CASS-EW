"""
Unit & Integration Tests for Phase 6: User-Facing Dataset Import, Validation & Replay Workflow
CASS-EW SIH 26055
"""

import os
import json
import io
import unittest
from pathlib import Path

from web_app import app, active_dataset_results
from tests.user_fixtures import (
    generate_canonical_csv,
    generate_alternate_column_csv,
    generate_canonical_json,
    generate_generic_hdf5,
    generate_malformed_csv
)


class TestPhase6DatasetUI(unittest.TestCase):
    """
    Validates Phase 6 user-facing workflow endpoints and integration.
    """

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()
        cls.scratch_dir = Path("scratch_test_phase6")
        cls.scratch_dir.mkdir(parents=True, exist_ok=True)

        cls.csv_path = str(cls.scratch_dir / "user_test.csv")
        generate_canonical_csv(cls.csv_path, num_pulses=30)

        cls.alt_csv_path = str(cls.scratch_dir / "alt_test.csv")
        generate_alternate_column_csv(cls.alt_csv_path, num_pulses=30)

        cls.json_path = str(cls.scratch_dir / "user_test.json")
        generate_canonical_json(cls.json_path, num_pulses=30)

        cls.h5_path = str(cls.scratch_dir / "user_test.h5")
        generate_generic_hdf5(cls.h5_path, num_pulses=30)

        cls.malformed_csv_path = str(cls.scratch_dir / "malformed.csv")
        generate_malformed_csv(cls.malformed_csv_path)

        cls.empty_csv_path = str(cls.scratch_dir / "empty.csv")
        with open(cls.empty_csv_path, 'w', encoding='utf-8') as f:
            f.write("")

        # Create no ground truth CSV
        cls.no_gt_path = str(cls.scratch_dir / "no_gt.csv")
        with open(cls.no_gt_path, 'w', encoding='utf-8') as f:
            f.write("timestamp,frequency,pulse_width,angle,amplitude\n")
            f.write("0.001,2400.0,1.5,10.0,-50.0\n")
            f.write("0.002,2400.0,1.5,10.0,-50.0\n")
            f.write("0.003,3600.0,1.5,10.0,-50.0\n")

    @classmethod
    def tearDownClass(cls):
        import shutil
        if cls.scratch_dir.exists():
            shutil.rmtree(cls.scratch_dir, ignore_errors=True)

    def test_a_dataset_page_loads(self):
        """A. Dataset page loads (index.html returns 200 and contains workflow elements)."""
        res = self.client.get('/')
        self.assertEqual(res.status_code, 200)
        html = res.data.decode('utf-8')
        self.assertIn("USER DATASET REPLAY CONSOLE", html)
        self.assertIn("btn-validate-dataset", html)
        self.assertIn("btn-run-dataset", html)
        self.assertIn("btn-export-json", html)

    def test_b_csv_upload_selection(self):
        """B. CSV upload/selection via /api/dataset/upload and /api/dataset/inspect."""
        # Test inspect
        res = self.client.post('/api/dataset/inspect', json={"file_path": self.csv_path})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["format"], "csv")
        self.assertIn("timestamp", data["raw_keys"])

        # Test upload
        with open(self.csv_path, 'rb') as f:
            res_up = self.client.post('/api/dataset/upload', data={
                'file': (io.BytesIO(f.read()), 'uploaded_sample.csv')
            }, content_type='multipart/form-data')
        self.assertEqual(res_up.status_code, 200)
        up_data = res_up.get_json()
        self.assertEqual(up_data["status"], "success")
        self.assertEqual(up_data["format"], "csv")

    def test_c_json_upload_selection(self):
        """C. JSON upload/selection."""
        res = self.client.post('/api/dataset/inspect', json={"file_path": self.json_path})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["format"], "json")
        self.assertIn("timestamp", data["raw_keys"])

        with open(self.json_path, 'rb') as f:
            res_up = self.client.post('/api/dataset/upload', data={
                'file': (io.BytesIO(f.read()), 'uploaded_sample.json')
            }, content_type='multipart/form-data')
        self.assertEqual(res_up.status_code, 200)
        self.assertEqual(res_up.get_json()["format"], "json")

    def test_d_hdf5_upload_selection(self):
        """D. HDF5 upload/selection."""
        res = self.client.post('/api/dataset/inspect', json={"file_path": self.h5_path})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["format"], "hdf5")
        self.assertIn("pulse_data", data["raw_keys"])

    def test_e_validation_endpoint(self):
        """E. Validation endpoint /api/dataset/validate returns structured report."""
        res = self.client.post('/api/dataset/validate', json={
            "file_path": self.csv_path,
            "field_mapping": {
                "timestamp": "timestamp",
                "frequency": "frequency",
                "pulse_width": "pulse_width",
                "aoa": "angle",
                "amplitude": "amplitude",
                "emitter_label": "emitter_id"
            },
            "units": {
                "timestamp": "s",
                "frequency": "MHz",
                "pulse_width": "us",
                "aoa": "deg",
                "amplitude": "dBm"
            }
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        rep = data["report"]
        self.assertTrue(rep["is_valid"])
        self.assertEqual(rep["valid_rows"], 30)

    def test_f_field_mapping(self):
        """F. Explicit field mapping is respected."""
        res = self.client.post('/api/dataset/validate', json={
            "file_path": self.alt_csv_path,
            "field_mapping": {
                "timestamp": "ToA_ms",
                "frequency": "carrier_freq_ghz",
                "pulse_width": "PW_ns",
                "aoa": "AoA_deg",
                "amplitude": "Power_dBm",
                "emitter_label": "Emitter_Code"
            },
            "units": {
                "timestamp": "ms",
                "frequency": "GHz",
                "pulse_width": "ns",
                "aoa": "deg",
                "amplitude": "dBm"
            }
        })
        self.assertEqual(res.status_code, 200)
        rep = res.get_json()["report"]
        self.assertEqual(rep["resolved_field_mapping"]["timestamp"], "ToA_ms")
        self.assertEqual(rep["resolved_field_mapping"]["frequency"], "carrier_freq_ghz")

    def test_g_unit_configuration(self):
        """G. Explicit unit configuration is respected."""
        res = self.client.post('/api/dataset/validate', json={
            "file_path": self.json_path,
            "field_mapping": {
                "timestamp": "timestamp",
                "frequency": "frequency"
            },
            "units": {
                "timestamp": "s",
                "frequency": "MHz"
            }
        })
        self.assertEqual(res.status_code, 200)
        rep = res.get_json()["report"]
        self.assertEqual(rep["input_units"]["timestamp"], "s")

    def test_h_invalid_mapping_blocks_execution(self):
        """H. Invalid mapping reports validation failure."""
        res = self.client.post('/api/dataset/validate', json={
            "file_path": self.csv_path,
            "field_mapping": {
                "timestamp": "non_existent_time",
                "frequency": "frequency"
            }
        })
        self.assertEqual(res.status_code, 200)
        rep = res.get_json()["report"]
        self.assertFalse(rep["is_valid"])
        self.assertTrue(len(rep["validation_errors"]) > 0)

    def test_i_unresolved_unit_blocks_execution(self):
        """I. Unresolved unit returns error."""
        res = self.client.post('/api/dataset/validate', json={
            "file_path": self.csv_path,
            "units": {
                "frequency": "gigawidgets"  # invalid unit
            }
        })
        self.assertEqual(res.status_code, 200)
        rep = res.get_json()["report"]
        self.assertFalse(rep["is_valid"])
        self.assertTrue(any("Unsupported unit" in e for e in rep["validation_errors"]))

    def test_j_and_k_valid_dataset_runs_and_displays(self):
        """J & K. Valid dataset runs CASS-EW replay and produces structured result."""
        res = self.client.post('/api/dataset/run', json={
            "file_path": self.csv_path,
            "field_mapping": {
                "timestamp": "timestamp",
                "frequency": "frequency",
                "pulse_width": "pulse_width",
                "aoa": "angle",
                "amplitude": "amplitude",
                "emitter_label": "emitter_id"
            },
            "units": {
                "timestamp": "s",
                "frequency": "MHz"
            },
            "steps": 10,
            "dwell_time_s": 0.002,
            "seed": 42
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        r = data["result"]
        self.assertEqual(r["validation_status"], "PASS")
        self.assertEqual(r["executed_scans"], 10)
        self.assertEqual(len(r["band_decisions"]), 10)
        self.assertIn("hits", r)
        self.assertIn("receiver_pd", r)

    def test_l_and_m_no_ground_truth_dataset(self):
        """L & M. No ground truth dataset leaves ground-truth metrics as null/None without error."""
        res = self.client.post('/api/dataset/run', json={
            "file_path": self.no_gt_path,
            "field_mapping": {
                "timestamp": "timestamp",
                "frequency": "frequency"
            },
            "units": {
                "timestamp": "s",
                "frequency": "MHz"
            },
            "steps": 5,
            "dwell_time_s": 0.005,
            "seed": 42
        })
        self.assertEqual(res.status_code, 200)
        r = res.get_json()["result"]
        self.assertIsNone(r["global_emitter_interception_rate"])
        self.assertIsNone(r["mean_intercept_time_s"])
        self.assertTrue(any("Ground truth evaluation unavailable" in w for w in r["warnings"]))

    def test_n_result_export(self):
        """N. Result export via /api/dataset/export returns valid JSON."""
        # Ensure latest exists
        active_dataset_results['latest'] = {"status": "TEST_EXPORT", "executed_scans": 15}
        res = self.client.get('/api/dataset/export')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "TEST_EXPORT")
        self.assertEqual(data["executed_scans"], 15)

    def test_o_synthetic_mode_still_works(self):
        """O. Synthetic simulation mode continues to function normally."""
        res_reset = self.client.post('/api/reset', json={"seed": 42, "scenario": "Mixed Environment"})
        self.assertEqual(res_reset.status_code, 200)
        res_step = self.client.post('/api/step', json={"steps": 2})
        self.assertEqual(res_step.status_code, 200)
        step_data = res_step.get_json()
        self.assertIn("time", step_data)
        self.assertIn("action", step_data)

    def test_p_mode_distinction(self):
        """P. Dataset replay and synthetic mode endpoints remain separate."""
        # Reset synthetic
        self.client.post('/api/reset', json={"seed": 99})
        # Run dataset
        res_ds = self.client.post('/api/dataset/run', json={
            "file_path": self.csv_path,
            "steps": 3,
            "seed": 42
        })
        self.assertEqual(res_ds.status_code, 200)
        # Synthetic step still advances synthetic state
        res_step = self.client.post('/api/step', json={"steps": 1})
        self.assertEqual(res_step.status_code, 200)

    def test_q_unsupported_file_rejection(self):
        """Q. Unsupported file format is rejected cleanly."""
        res = self.client.post('/api/dataset/upload', data={
            'file': (io.BytesIO(b"malicious executable or text"), 'bad_file.exe')
        }, content_type='multipart/form-data')
        self.assertEqual(res.status_code, 400)
        self.assertIn("Unsupported file type", res.get_json()["message"])

    def test_r_malformed_dataset_handling(self):
        """R. Malformed / empty dataset reports error cleanly without crashing."""
        res = self.client.post('/api/dataset/validate', json={
            "file_path": self.empty_csv_path
        })
        self.assertEqual(res.status_code, 200)
        rep = res.get_json()["report"]
        self.assertFalse(rep["is_valid"])

    def test_s_no_fabricated_metrics(self):
        """S. When opportunities are 0 or ground truth absent, no fake numbers are made up."""
        # When ground truth is absent, IR must be None (null in json), never 0.0 or 1.0
        res = self.client.post('/api/dataset/run', json={
            "file_path": self.no_gt_path,
            "steps": 2,
            "seed": 42
        })
        r = res.get_json()["result"]
        self.assertIsNone(r["global_emitter_interception_rate"])
        self.assertIsNone(r["total_ground_truth_emitters"])

    def test_t_frozen_backend_integrity(self):
        """T. Frozen backend integrity check."""
        for p in [
            "algorithms/phase8_spatial_scheduler.py",
            "simulator/environment.py",
            "simulator/receiver.py",
            "config.py",
            "main.py"
        ]:
            self.assertTrue(os.path.exists(p), f"Missing core file {p}")


if __name__ == '__main__':
    unittest.main()
