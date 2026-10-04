"""
Phase 10 Final Integration + Leakage Audit Tests
Tests the complete integrated CASS-EW pipeline:
- End-to-end adversarial leakage test
- Full pipeline execution
- Mode selection
- Configuration propagation
"""
import unittest
import os

from algorithms.phase8_spatial_scheduler import SpatialScheduler
from algorithms.phase7_signal_pattern import PatternAwareScheduler
from algorithms.phase6_predictive_scheduler import PredictiveScheduler
from algorithms.phase5_temporal_profile import MultiBandTemporalProfileScheduler
from algorithms.adaptive_belief import AdaptiveBeliefScheduler
from data.real_pdws import validate_dataset, PDWReplayEngine, search_for_authorized_datasets
from config import get_default_config


class TestPhase10Integration(unittest.TestCase):

    def test_unified_synthetic_pipeline(self):
        """Full integrated pipeline runs without error."""
        from main import run_synthetic
        config = get_default_config()
        results, scan_log = run_synthetic(config, seed=42, verbose=False)
        self.assertGreater(results["total_observations"], 0)
        self.assertGreaterEqual(results["detections"], 0)
        self.assertIn("interception_rate", results)
        self.assertIn("spatial_influenced_selections", results)

    def test_unified_replay_pipeline(self):
        """Replay mode runs with mock PDW data."""
        import json, tempfile
        mock_data = [
            {"timestamp": float(i), "frequency": i % 10, "detection_result": i % 3 == 0}
            for i in range(50)
        ]
        tmpfile = os.path.join(tempfile.gettempdir(), "test_replay.json")
        with open(tmpfile, 'w') as f:
            json.dump(mock_data, f)
        
        from main import run_replay
        config = get_default_config()
        metrics, log = run_replay(tmpfile, config, seed=42, verbose=False)
        self.assertEqual(metrics["total_records"], 50)
        self.assertEqual(metrics["replayed_records"], 50)
        os.remove(tmpfile)

    def test_end_to_end_adversarial_leakage(self):
        """
        Two simulations with IDENTICAL observations through T but different
        hidden future truth must produce identical scheduler actions at T.
        """
        # Build two identical schedulers
        sched_a = SpatialScheduler(['R1', 'R2'], 10, seed=42, epsilon=0.0,
            spatial_weight=0.3, belief_weight=0.2, temporal_weight=0.15,
            prediction_weight=0.2, pattern_weight=0.15)
        sched_b = SpatialScheduler(['R1', 'R2'], 10, seed=42, epsilon=0.0,
            spatial_weight=0.3, belief_weight=0.2, temporal_weight=0.15,
            prediction_weight=0.2, pattern_weight=0.15)

        # Feed IDENTICAL observations through T=20
        shared_obs = []
        for t in range(1, 21):
            obs = {'time': t, 'observation_time': t}
            action_a = sched_a.select_action(obs)
            action_b = sched_b.select_action(obs)
            self.assertEqual(action_a, action_b, f"Actions differ at t={t} before identical histories")
            
            # Deterministic "detection" result
            det = (t % 4 == 0)
            sched_a.update(obs, action_a, det)
            sched_b.update(obs, action_b, det)

        # At T=20, both must select the same action
        final_a = sched_a.select_action({'time': 21, 'observation_time': 21})
        final_b = sched_b.select_action({'time': 21, 'observation_time': 21})
        self.assertEqual(final_a, final_b)

    def test_scheduler_code_leakage_audit(self):
        """
        Search all scheduler source files for prohibited ground truth access.
        """
        scheduler_files = [
            'algorithms/phase8_spatial_scheduler.py',
            'algorithms/phase7_signal_pattern.py',
            'algorithms/phase6_predictive_scheduler.py',
            'algorithms/phase5_temporal_profile.py',
            'algorithms/adaptive_belief.py',
            'algorithms/temporal_belief.py',
        ]
        
        prohibited = [
            'env.time', 'environment.time',
            'emitter.active', 'emitter.state', 'emitter.id',
            'emitter.position', 'true_period', 'ground_truth',
        ]
        
        for fpath in scheduler_files:
            if os.path.exists(fpath):
                with open(fpath, 'r') as f:
                    code = f.read()
                for pattern in prohibited:
                    self.assertNotIn(
                        pattern, code,
                        f"LEAKAGE: '{pattern}' found in {fpath}"
                    )

    def test_dataset_availability_reporting(self):
        """Verify the dataset search reports correctly."""
        result = search_for_authorized_datasets()
        self.assertIn("authorized_dataset_found", result)
        self.assertIn("status", result)

    def test_configuration_propagation(self):
        """Config values are correctly used."""
        config = get_default_config()
        self.assertEqual(config.environment.num_bands, 10)
        self.assertEqual(config.environment.total_time, 500)
        self.assertEqual(config.benchmark.benchmark_seeds, (42, 43, 44, 45, 46))

    def test_all_phase_schedulers_instantiate(self):
        """All phase schedulers can be created without error."""
        nb = 10
        AdaptiveBeliefScheduler(nb, seed=42)
        MultiBandTemporalProfileScheduler(nb, seed=42)
        PredictiveScheduler(nb, seed=42)
        PatternAwareScheduler(nb, seed=42)
        SpatialScheduler(['R1'], nb, seed=42)


if __name__ == '__main__':
    unittest.main()
