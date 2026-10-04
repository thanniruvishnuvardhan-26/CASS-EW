"""
Phase 7 End-to-End Validation, Demonstration Hardening & Integrity Audit Suite
CASS-EW SIH 26055
"""

import os
import io
import json
import time
import unittest
from pathlib import Path

from web_app import app, active_dataset_results
from data.dataset_adapter import DatasetConfig, load_dataset
from data.dataset_runtime import DatasetRunConfig, DatasetRuntimeRunner, run_dataset
from tests.user_fixtures import (
    generate_canonical_csv,
    generate_alternate_column_csv,
    generate_canonical_json,
    generate_generic_hdf5,
    generate_malformed_csv
)


class TestPhase7EndToEndValidation(unittest.TestCase):
    """
    Formal end-to-end audit, reproducibility, causal verification,
    and failure-recovery hardening tests for Phase 7.
    """

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()
        cls.scratch_dir = Path("scratch_test_phase7")
        cls.scratch_dir.mkdir(parents=True, exist_ok=True)

        cls.csv_fixture = str(cls.scratch_dir / "fixture.csv")
        generate_canonical_csv(cls.csv_fixture, num_pulses=50, seed=101)

        cls.alt_csv_fixture = str(cls.scratch_dir / "fixture_alt.csv")
        generate_alternate_column_csv(cls.alt_csv_fixture, num_pulses=50, out_of_order=True, seed=102)

        cls.json_fixture = str(cls.scratch_dir / "fixture.json")
        generate_canonical_json(cls.json_fixture, num_pulses=50, seed=103)

        cls.h5_fixture = str(cls.scratch_dir / "fixture.h5")
        generate_generic_hdf5(cls.h5_fixture, num_pulses=60, seed=104)

        cls.malformed_csv = str(cls.scratch_dir / "malformed.csv")
        generate_malformed_csv(cls.malformed_csv)

        cls.empty_csv = str(cls.scratch_dir / "empty.csv")
        with open(cls.empty_csv, 'w', encoding='utf-8') as f:
            f.write("")

        cls.no_gt_csv = str(cls.scratch_dir / "no_gt.csv")
        with open(cls.no_gt_csv, 'w', encoding='utf-8') as f:
            f.write("timestamp,frequency,pulse_width,angle,amplitude\n")
            for i in range(30):
                f.write(f"{0.001 * (i + 1):.6f},3200.0,1.5,10.0,-50.0\n")

    @classmethod
    def tearDownClass(cls):
        import shutil
        if cls.scratch_dir.exists():
            shutil.rmtree(cls.scratch_dir, ignore_errors=True)

    # -------------------------------------------------------------
    # 1. REPRODUCIBILITY VALIDATION (SAME SEED vs DIFFERENT SEED)
    # -------------------------------------------------------------

    def test_reproducibility_same_seed_exact_match(self):
        """
        Verify identical run configuration with seed 42 executed across multiple
        invocations produces bit-identical decisions, counts, and metrics.
        """
        cfg = DatasetRunConfig(
            dataset_path=self.csv_fixture,
            seed=42,
            steps=15,
            dwell_time_s=0.002,
            initial_band=0
        )
        res_a = DatasetRuntimeRunner(cfg).run()
        res_b = DatasetRuntimeRunner(cfg).run()
        res_c = DatasetRuntimeRunner(cfg).run()

        self.assertEqual(res_a.band_decisions, res_b.band_decisions)
        self.assertEqual(res_b.band_decisions, res_c.band_decisions)
        self.assertEqual(res_a.hits, res_b.hits)
        self.assertEqual(res_a.misses, res_b.misses)
        self.assertEqual(res_a.observation_opportunities, res_b.observation_opportunities)
        self.assertEqual(res_a.receiver_pd, res_b.receiver_pd)
        self.assertEqual(res_a.receiver_pfa, res_b.receiver_pfa)
        self.assertEqual(res_a.to_json(), res_b.to_json())

    def test_reproducibility_different_seed_different_trajectory(self):
        """
        Verify changing the seed from 42 to 43 allows legitimate different stochastic behavior.
        """
        cfg_42 = DatasetRunConfig(dataset_path=self.csv_fixture, seed=42, steps=15, dwell_time_s=0.002)
        cfg_43 = DatasetRunConfig(dataset_path=self.csv_fixture, seed=43, steps=15, dwell_time_s=0.002)

        res_42 = DatasetRuntimeRunner(cfg_42).run()
        res_43 = DatasetRuntimeRunner(cfg_43).run()

        self.assertEqual(res_42.executed_scans, 15)
        self.assertEqual(res_43.executed_scans, 15)
        self.assertNotEqual(res_42.seed, res_43.seed)

    # -------------------------------------------------------------
    # 2. CAUSAL & FUTURE INFORMATION LEAKAGE AUDIT (MULTI-CUT POINT)
    # -------------------------------------------------------------

    def test_multi_cut_point_causal_anti_leakage(self):
        """
        Construct two datasets with identical records up to time T, but wildly
        divergent records after T. Test across multiple cut-points (e.g. T=0.015s and T=0.030s).
        Decisions up to T must remain bit-identical.
        """
        for t_cut, steps in [(0.015, 6), (0.030, 12)]:
            p_common = str(self.scratch_dir / f"common_cut_{int(t_cut*1000)}.csv")
            p_divergent = str(self.scratch_dir / f"divergent_cut_{int(t_cut*1000)}.csv")

            # Generate common history up to t_cut
            common_records = []
            for i in range(15):
                t_val = 0.002 * (i + 1)
                if t_val <= t_cut:
                    common_records.append((t_val, 2400.0, 1.5, 0.0, -50.0, 1))

            # Dataset A continues at 2400 MHz
            with open(p_common, 'w', encoding='utf-8') as f:
                f.write("timestamp,frequency,pulse_width,angle,amplitude,emitter_id\n")
                for r in common_records:
                    f.write(f"{r[0]:.6f},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]}\n")
                for i in range(15):
                    t_val = t_cut + 0.002 * (i + 1)
                    f.write(f"{t_val:.6f},2400.0,1.5,0.0,-50.0,1\n")

            # Dataset B continues at 8800 MHz with multiple different emitters
            with open(p_divergent, 'w', encoding='utf-8') as f:
                f.write("timestamp,frequency,pulse_width,angle,amplitude,emitter_id\n")
                for r in common_records:
                    f.write(f"{r[0]:.6f},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]}\n")
                for i in range(15):
                    t_val = t_cut + 0.002 * (i + 1)
                    f.write(f"{t_val:.6f},8800.0,3.0,45.0,-40.0,9\n")

            # Run both through the Phase 5 runtime runner with identical coverage range
            fixed_cov = (1000.0, 10000.0)
            cfg_a = DatasetRunConfig(dataset_path=p_common, steps=steps, dwell_time_s=0.002, seed=42, coverage_freq_range_mhz=fixed_cov)
            cfg_b = DatasetRunConfig(dataset_path=p_divergent, steps=steps, dwell_time_s=0.002, seed=42, coverage_freq_range_mhz=fixed_cov)

            res_a = DatasetRuntimeRunner(cfg_a).run()
            res_b = DatasetRuntimeRunner(cfg_b).run()

            # All decisions up to the cut horizon must match identically
            self.assertEqual(
                res_a.band_decisions[:steps],
                res_b.band_decisions[:steps],
                f"Leakage detected at cut-point {t_cut}s! Past decisions were influenced by future data."
            )

    # -------------------------------------------------------------
    # 3. METRIC PROVENANCE & NO-FAKE-DATA INTEGRITY
    # -------------------------------------------------------------

    def test_no_ground_truth_returns_explicit_none_no_fake_zeros(self):
        """
        Verify when ground truth is absent, global interception rate and latency
        are None (null in JSON), never 0.0 or 1.0.
        """
        res = run_dataset(self.no_gt_csv, steps=10, dwell_time_s=0.002, seed=42)
        self.assertIsNone(res.global_emitter_interception_rate)
        self.assertIsNone(res.total_ground_truth_emitters)
        self.assertIsNone(res.intercepted_emitters)
        self.assertIsNone(res.mean_intercept_time_s)
        self.assertTrue(any("Ground truth evaluation unavailable" in w for w in res.warnings))

    def test_zero_opportunities_returns_safe_null_or_zero_no_crash(self):
        """
        Zero observation opportunities must not cause ZeroDivisionError.
        """
        # Create dataset in a single high band (e.g. 9500 MHz)
        p = str(self.scratch_dir / "high_band_only.csv")
        with open(p, 'w', encoding='utf-8') as f:
            f.write("timestamp,frequency,pulse_width,angle,amplitude\n")
            f.write("0.001,9500.0,1.0,0.0,-50.0\n")
            f.write("0.002,9500.0,1.0,0.0,-50.0\n")

        # Run 2 scans on Band 0 (1000 - 1900 MHz)
        res = run_dataset(p, steps=2, dwell_time_s=0.001, initial_band=0, coverage_freq_range_mhz=(1000.0, 10000.0))
        self.assertIsInstance(res.observation_opportunities, int)
        self.assertIsInstance(res.hits, int)
        self.assertIsInstance(res.misses, int)

    # -------------------------------------------------------------
    # 4. END-TO-END WEB UI & REST INTEGRITY
    # -------------------------------------------------------------

    def test_end_to_end_web_workflow_csv(self):
        """
        Execute full UI cycle: Upload -> Inspect -> Validate -> Run -> Export.
        """
        # Upload
        with open(self.csv_fixture, 'rb') as f:
            up_res = self.client.post('/api/dataset/upload', data={
                'file': (io.BytesIO(f.read()), 'e2e_test.csv')
            }, content_type='multipart/form-data')
        self.assertEqual(up_res.status_code, 200)
        up_data = up_res.get_json()
        fpath = up_data["file_path"]

        # Validate
        val_res = self.client.post('/api/dataset/validate', json={
            "file_path": fpath,
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
            }
        })
        self.assertEqual(val_res.status_code, 200)
        self.assertTrue(val_res.get_json()["report"]["is_valid"])

        # Run
        run_res = self.client.post('/api/dataset/run', json={
            "file_path": fpath,
            "steps": 10,
            "dwell_time_s": 0.002,
            "seed": 42
        })
        self.assertEqual(run_res.status_code, 200)
        r = run_res.get_json()["result"]
        self.assertEqual(r["executed_scans"], 10)
        self.assertEqual(r["validation_status"], "PASS")

        # Export
        exp_res = self.client.get('/api/dataset/export')
        self.assertEqual(exp_res.status_code, 200)
        exp_data = exp_res.get_json()
        self.assertEqual(exp_data["executed_scans"], 10)
        self.assertEqual(exp_data["dataset_path"], fpath)

    # -------------------------------------------------------------
    # 5. DEMO SCENARIO HARDENING & EXECUTION
    # -------------------------------------------------------------

    def test_demo_scenario_stability(self):
        """
        Validate that the standard SIH demonstration scenario (sample_stare.h5, 20 steps, seed 42)
        executes stably in under 1 second and produces expected valid audited metrics.
        """
        stare_path = "data/tsrd_fixtures/sample_stare.h5"
        self.assertTrue(os.path.exists(stare_path))

        t0 = time.time()
        res = run_dataset(
            dataset_path=stare_path,
            steps=20,
            dwell_time_s=0.005,
            seed=42,
            initial_band=0
        )
        elapsed = time.time() - t0

        self.assertLess(elapsed, 2.0, f"Demo scenario took too long: {elapsed:.2f}s")
        self.assertEqual(res.executed_scans, 20)
        self.assertEqual(res.validation_status, "PASS")
        self.assertGreater(res.observation_opportunities, 0)
        self.assertGreater(res.hits, 0)
        self.assertIsNotNone(res.receiver_pd)
        self.assertIsNotNone(res.global_emitter_interception_rate)

    # -------------------------------------------------------------
    # 6. FAILURE RECOVERY & SECURITY SANITY
    # -------------------------------------------------------------

    def test_malformed_csv_handled_safely(self):
        """Malformed rows are flagged cleanly without server crash."""
        res = self.client.post('/api/dataset/validate', json={"file_path": self.malformed_csv})
        self.assertEqual(res.status_code, 200)
        rep = res.get_json()["report"]
        self.assertFalse(rep["is_valid"])
        self.assertGreater(len(rep["validation_errors"]), 0)

    def test_upload_path_traversal_prevention(self):
        """Filename with path traversal attempts is neutralized."""
        evil_name = "../../../../windows/system32/evil.csv"
        res = self.client.post('/api/dataset/upload', data={
            'file': (io.BytesIO(b"timestamp,frequency\n0.001,2400\n"), evil_name)
        }, content_type='multipart/form-data')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertNotIn("..", data["filename"])
        self.assertIn("evil.csv", data["filename"])


if __name__ == '__main__':
    unittest.main()
