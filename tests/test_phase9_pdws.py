"""
Phase 9 Test Suite: PDW Data Adapter
Tests: valid parsing, malformed rejection, timestamp ordering, duplicate handling,
missing values, unit normalization, deterministic replay, future-record isolation,
scheduler observation boundary, no hidden future information, empty dataset,
single-record dataset, large replay fixture, receiver ID handling
"""
import unittest
import json
import os
import tempfile

from data.real_pdws import (
    validate_pdw_record, validate_dataset, normalize_to_observation,
    PDWRecord, PDWValidationReport, PDWReplayEngine,
    load_pdw_json, search_for_authorized_datasets
)
from algorithms.adaptive_belief import AdaptiveBeliefScheduler


class TestPhase9PDWs(unittest.TestCase):

    # 1. Valid PDW parsing
    def test_valid_pdw_parsing(self):
        record = {"timestamp": 1.0, "frequency": 3, "amplitude": -40.0,
                  "receiver_id": "R1", "detection_result": True}
        report = PDWValidationReport()
        pdw = validate_pdw_record(record, 0, report)
        self.assertIsNotNone(pdw)
        self.assertEqual(pdw.timestamp, 1.0)
        self.assertEqual(pdw.frequency, 3)
        self.assertEqual(pdw.receiver_id, "R1")
        self.assertTrue(pdw.detection_result)
        self.assertEqual(report.accepted, 1)

    # 2. Malformed record rejection
    def test_malformed_record_rejection(self):
        # Missing timestamp
        report = PDWValidationReport()
        pdw = validate_pdw_record({"frequency": 3}, 0, report)
        self.assertIsNone(pdw)
        self.assertEqual(report.rejected, 1)

        # Missing frequency
        report2 = PDWValidationReport()
        pdw2 = validate_pdw_record({"timestamp": 1.0}, 0, report2)
        self.assertIsNone(pdw2)

        # Wrong type
        report3 = PDWValidationReport()
        pdw3 = validate_pdw_record({"timestamp": "abc", "frequency": 3}, 0, report3)
        self.assertIsNone(pdw3)

    # 3. Timestamp ordering
    def test_timestamp_ordering(self):
        records = [
            {"timestamp": 1.0, "frequency": 3},
            {"timestamp": 3.0, "frequency": 5},
            {"timestamp": 2.0, "frequency": 1},  # out of order
        ]
        valid, report = validate_dataset(records)
        self.assertEqual(len(valid), 2)  # third rejected for ordering
        self.assertEqual(report.rejected, 1)

    # 4. Duplicate handling
    def test_duplicate_handling(self):
        records = [
            {"timestamp": 1.0, "frequency": 3, "receiver_id": "R0"},
            {"timestamp": 1.0, "frequency": 3, "receiver_id": "R0"},  # exact duplicate
        ]
        valid, report = validate_dataset(records)
        self.assertEqual(len(valid), 1)
        self.assertEqual(report.rejected, 1)

    # 5. Missing values
    def test_missing_values(self):
        # Optional fields can be None/missing
        record = {"timestamp": 1.0, "frequency": 3}
        report = PDWValidationReport()
        pdw = validate_pdw_record(record, 0, report)
        self.assertIsNotNone(pdw)
        self.assertIsNone(pdw.bandwidth)
        self.assertIsNone(pdw.pulse_width)
        self.assertIsNone(pdw.amplitude)
        self.assertEqual(pdw.receiver_id, "R0")

    # 6. Unit normalization
    def test_unit_normalization(self):
        pdw = PDWRecord({"timestamp": 5.0, "frequency": 3, "pulse_width": 0.5})
        obs = normalize_to_observation(pdw, num_bands=10)
        self.assertEqual(obs["tuned_band"], 3)
        self.assertEqual(obs["band"], 3)
        self.assertEqual(obs["observation_time"], 5.0)
        self.assertEqual(obs["effective_duration"], 0.5)

    # 7. Deterministic replay
    def test_deterministic_replay(self):
        records = [
            {"timestamp": 1.0, "frequency": 2, "detection_result": True},
            {"timestamp": 2.0, "frequency": 5, "detection_result": False},
            {"timestamp": 3.0, "frequency": 2, "detection_result": True},
        ]
        valid, _ = validate_dataset(records)

        sched1 = AdaptiveBeliefScheduler(num_bands=10, seed=42, epsilon=0.0)
        engine1 = PDWReplayEngine(valid, sched1, num_bands=10)
        log1 = engine1.run_full_replay()

        sched2 = AdaptiveBeliefScheduler(num_bands=10, seed=42, epsilon=0.0)
        engine2 = PDWReplayEngine(valid, sched2, num_bands=10)
        log2 = engine2.run_full_replay()

        # Deterministic: identical action sequences
        for e1, e2 in zip(log1, log2):
            self.assertEqual(e1["scheduler_action"], e2["scheduler_action"])

    # 8. Future-record isolation
    def test_future_record_isolation(self):
        """The replay engine feeds records one at a time in order.
        After processing record i, the scheduler has NOT seen record i+1."""
        records = [
            {"timestamp": 1.0, "frequency": 2, "detection_result": True},
            {"timestamp": 2.0, "frequency": 5, "detection_result": False},
            {"timestamp": 3.0, "frequency": 2, "detection_result": True},
        ]
        valid, _ = validate_dataset(records)
        sched = AdaptiveBeliefScheduler(num_bands=10, seed=42, epsilon=0.0)
        engine = PDWReplayEngine(valid, sched, num_bands=10)

        # Step once
        engine.step()
        self.assertEqual(engine.replay_index, 1)
        # Scheduler has seen only record 0
        self.assertEqual(len(engine.action_log), 1)

    # 9. Scheduler observation boundary
    def test_scheduler_observation_boundary(self):
        """Scheduler must not be aware it's in replay mode."""
        records = [
            {"timestamp": 1.0, "frequency": 2, "detection_result": True},
        ]
        valid, _ = validate_dataset(records)
        sched = AdaptiveBeliefScheduler(num_bands=10, seed=42, epsilon=0.0)
        engine = PDWReplayEngine(valid, sched, num_bands=10)
        engine.step()
        # The observation format matches the synthetic observation contract
        obs = engine.action_log[0]["observation"]
        self.assertIn("observation_time", obs)
        self.assertIn("tuned_band", obs)
        self.assertIn("detection_result", obs)
        self.assertIn("effective_duration", obs)

    # 10. No hidden future information
    def test_no_hidden_future_information(self):
        """PDWReplayEngine does not expose future records to scheduler."""
        engine_code_path = 'data/real_pdws.py'
        with open(engine_code_path, 'r') as f:
            code = f.read()
        # The replay engine should not expose all records to the scheduler
        self.assertNotIn('scheduler.records', code)
        self.assertNotIn('scheduler.all_data', code)

    # 11. Empty dataset
    def test_empty_dataset(self):
        valid, report = validate_dataset([])
        self.assertEqual(len(valid), 0)
        self.assertEqual(report.accepted, 0)
        self.assertEqual(report.rejected, 0)

    # 12. Single-record dataset
    def test_single_record_dataset(self):
        records = [{"timestamp": 1.0, "frequency": 3, "detection_result": True}]
        valid, report = validate_dataset(records)
        self.assertEqual(len(valid), 1)
        sched = AdaptiveBeliefScheduler(num_bands=10, seed=42)
        engine = PDWReplayEngine(valid, sched, num_bands=10)
        log = engine.run_full_replay()
        self.assertEqual(len(log), 1)

    # 13. Large replay fixture
    def test_large_replay_fixture(self):
        records = [
            {"timestamp": float(i), "frequency": i % 10, "detection_result": i % 3 == 0}
            for i in range(500)
        ]
        valid, report = validate_dataset(records)
        self.assertEqual(len(valid), 500)
        sched = AdaptiveBeliefScheduler(num_bands=10, seed=42)
        engine = PDWReplayEngine(valid, sched, num_bands=10)
        log = engine.run_full_replay()
        self.assertEqual(len(log), 500)
        metrics = engine.get_replay_metrics()
        self.assertEqual(metrics["total_records"], 500)
        self.assertEqual(metrics["replayed_records"], 500)

    # 14. Receiver ID handling
    def test_receiver_id_handling(self):
        records = [
            {"timestamp": 1.0, "frequency": 2, "receiver_id": "RX_ALPHA", "detection_result": True},
            {"timestamp": 2.0, "frequency": 5, "receiver_id": "RX_BETA", "detection_result": False},
        ]
        valid, _ = validate_dataset(records)
        self.assertEqual(valid[0].receiver_id, "RX_ALPHA")
        self.assertEqual(valid[1].receiver_id, "RX_BETA")

    # Negative value rejection
    def test_negative_value_rejection(self):
        report = PDWValidationReport()
        pdw = validate_pdw_record({"timestamp": -1.0, "frequency": 3}, 0, report)
        self.assertIsNone(pdw)

        report2 = PDWValidationReport()
        pdw2 = validate_pdw_record({"timestamp": 1.0, "frequency": -5}, 0, report2)
        self.assertIsNone(pdw2)

    # Int-to-bool correction
    def test_int_to_bool_correction(self):
        report = PDWValidationReport()
        pdw = validate_pdw_record({"timestamp": 1.0, "frequency": 3, "detection_result": 1}, 0, report)
        self.assertIsNotNone(pdw)
        self.assertTrue(pdw.detection_result)
        self.assertEqual(report.corrected, 1)

    # JSON load test
    def test_json_load(self):
        data = [
            {"timestamp": 1.0, "frequency": 2},
            {"timestamp": 2.0, "frequency": 5},
        ]
        tmpfile = os.path.join(tempfile.gettempdir(), "test_pdw.json")
        with open(tmpfile, 'w') as f:
            json.dump(data, f)
        loaded = load_pdw_json(tmpfile)
        self.assertEqual(len(loaded), 2)
        os.remove(tmpfile)

    # Authorized dataset search
    def test_authorized_dataset_search(self):
        result = search_for_authorized_datasets()
        self.assertIn("authorized_dataset_found", result)
        self.assertIn("status", result)

    # Band mapping normalization
    def test_band_mapping(self):
        pdw = PDWRecord({"timestamp": 1.0, "frequency": 2400.0})
        mapping = {2400.0: 3}
        obs = normalize_to_observation(pdw, band_mapping=mapping, num_bands=10)
        self.assertEqual(obs["tuned_band"], 3)

    # Validation report summary
    def test_validation_report_summary(self):
        report = PDWValidationReport()
        report.accept()
        report.accept()
        report.reject("test reason", 5)
        summary = report.summary()
        self.assertEqual(summary["accepted"], 2)
        self.assertEqual(summary["rejected"], 1)


if __name__ == '__main__':
    unittest.main()
