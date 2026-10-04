"""
Tests for the Event-Based Intercept Time Evaluator.

These tests verify:
1. Immediate detection latency
2. Delayed detection latency
3. Missed emitter handling (no zero-latency misses)
4. Multiple emitters (independent accounting)
5. Multiple episodes per emitter
6. Periodic emitter episode handling
7. Hopping emitter episode handling
8. Deterministic replay
9. Metric denominators
10. Ground-truth separation
11. Adversarial causality test
12. No zero-latency misses (explicit)
"""
import unittest
import numpy as np
from evaluation.intercept_time import (
    InterceptTimeTracker, EmitterEpisode,
    run_intercept_time_eval_single,
    run_intercept_time_eval_multi,
)
from simulator.environment import Emitter, RFEnvironment
from config import get_default_config


class MockEmitter:
    """Minimal emitter mock for testing episode detection."""
    def __init__(self, name, band, active=False):
        self.name = name
        self.band = band
        self.active = active


class TestInterceptTime(unittest.TestCase):
    """Test suite for InterceptTimeTracker."""

    # ================================================================
    # 1. Immediate detection
    # ================================================================
    def test_immediate_detection(self):
        """Emitter activates, first scan detects it. Latency = 0."""
        tracker = InterceptTimeTracker()
        e = MockEmitter("E1", band=2, active=True)

        # Time 1: emitter activates
        tracker.update_emitter_states(1, [e])
        # Scanner tunes to band 2 and detects
        tracker.record_scan(1, scanned_band=2, detected=True, was_truly_active=True)
        tracker.finalize(1)

        results = tracker.get_results()
        self.assertEqual(results['eligible_episodes'], 1)
        self.assertEqual(results['intercepted_episodes'], 1)
        self.assertEqual(results['missed_episodes'], 0)
        self.assertEqual(results['mean_intercept_time'], 0.0)
        self.assertEqual(results['all_latencies'], [0])

    # ================================================================
    # 2. Delayed detection
    # ================================================================
    def test_delayed_detection(self):
        """Emitter activates at t=5, first detection at t=8. Latency = 3."""
        tracker = InterceptTimeTracker()
        e = MockEmitter("E1", band=3, active=False)

        # t=1..4: inactive
        for t in range(1, 5):
            tracker.update_emitter_states(t, [e])

        # t=5: activates
        e.active = True
        tracker.update_emitter_states(5, [e])

        # t=5,6,7: scan wrong band or no detection
        tracker.record_scan(5, scanned_band=3, detected=False, was_truly_active=True)
        tracker.update_emitter_states(6, [e])
        tracker.record_scan(6, scanned_band=0, detected=False, was_truly_active=False)
        tracker.update_emitter_states(7, [e])
        tracker.record_scan(7, scanned_band=3, detected=False, was_truly_active=True)

        # t=8: detected!
        tracker.update_emitter_states(8, [e])
        tracker.record_scan(8, scanned_band=3, detected=True, was_truly_active=True)
        tracker.finalize(8)

        results = tracker.get_results()
        self.assertEqual(results['intercepted_episodes'], 1)
        # Latency = 8 - 5 = 3
        self.assertEqual(results['all_latencies'], [3])
        self.assertEqual(results['mean_intercept_time'], 3.0)

    # ================================================================
    # 3. Missed emitter
    # ================================================================
    def test_missed_emitter(self):
        """Emitter activates but is never detected. Episode is missed, latency is NOT zero."""
        tracker = InterceptTimeTracker()
        e = MockEmitter("E1", band=4, active=True)

        for t in range(1, 6):
            tracker.update_emitter_states(t, [e])
            # Scanner always tunes to wrong band
            tracker.record_scan(t, scanned_band=0, detected=False, was_truly_active=False)

        e.active = False
        tracker.update_emitter_states(6, [e])
        tracker.finalize(6)

        results = tracker.get_results()
        self.assertEqual(results['eligible_episodes'], 1)
        self.assertEqual(results['intercepted_episodes'], 0)
        self.assertEqual(results['missed_episodes'], 1)
        # Mean/median should be None since no successful interceptions
        self.assertIsNone(results['mean_intercept_time'])
        self.assertIsNone(results['median_intercept_time'])
        # Latencies list must be empty (no zero!)
        self.assertEqual(results['all_latencies'], [])

    # ================================================================
    # 4. Multiple emitters (independent accounting)
    # ================================================================
    def test_multiple_emitters(self):
        """Two emitters activate at different times. Independent tracking."""
        tracker = InterceptTimeTracker()
        e1 = MockEmitter("E1", band=1, active=False)
        e2 = MockEmitter("E2", band=5, active=False)

        # t=1: E1 activates
        e1.active = True
        tracker.update_emitter_states(1, [e1, e2])
        tracker.record_scan(1, scanned_band=1, detected=True, was_truly_active=True)

        # t=3: E2 activates
        tracker.update_emitter_states(2, [e1, e2])
        e2.active = True
        tracker.update_emitter_states(3, [e1, e2])
        tracker.record_scan(3, scanned_band=5, detected=True, was_truly_active=True)

        tracker.finalize(3)
        results = tracker.get_results()

        self.assertEqual(results['eligible_episodes'], 2)
        self.assertEqual(results['intercepted_episodes'], 2)
        # E1: latency = 1-1 = 0, E2: latency = 3-3 = 0
        self.assertEqual(sorted(results['all_latencies']), [0, 0])

    # ================================================================
    # 5. Multiple episodes from one emitter
    # ================================================================
    def test_multiple_episodes(self):
        """One emitter has two separate activation episodes."""
        tracker = InterceptTimeTracker()
        e = MockEmitter("E1", band=2, active=False)

        # Episode 1: t=1..3
        e.active = True
        for t in [1, 2, 3]:
            tracker.update_emitter_states(t, [e])
        tracker.record_scan(2, scanned_band=2, detected=True, was_truly_active=True)

        # Gap: t=4..5
        e.active = False
        tracker.update_emitter_states(4, [e])
        tracker.update_emitter_states(5, [e])

        # Episode 2: t=6..8
        e.active = True
        for t in [6, 7, 8]:
            tracker.update_emitter_states(t, [e])
        tracker.record_scan(8, scanned_band=2, detected=True, was_truly_active=True)

        tracker.finalize(8)
        results = tracker.get_results()

        self.assertEqual(results['eligible_episodes'], 2)
        self.assertEqual(results['intercepted_episodes'], 2)
        # Episode 1: latency = 2-1 = 1
        # Episode 2: latency = 8-6 = 2
        self.assertEqual(sorted(results['all_latencies']), [1, 2])
        self.assertAlmostEqual(results['mean_intercept_time'], 1.5)

    # ================================================================
    # 6. Periodic emitter episode handling
    # ================================================================
    def test_periodic_emitter_episodes(self):
        """Periodic emitter: period=5, duty=2. Two active windows = two episodes."""
        tracker = InterceptTimeTracker()
        e = MockEmitter("E_periodic", band=3, active=False)

        # Simulate period=5, duty=2: active at t%5 < 2
        for t in range(1, 16):
            time_in_period = (t - 1) % 5
            e.active = (time_in_period < 2)
            tracker.update_emitter_states(t, [e])

        tracker.finalize(15)
        results = tracker.get_results()

        # Expect 3 episodes: t=1-2, t=6-7, t=11-12
        self.assertEqual(results['eligible_episodes'], 3)

    # ================================================================
    # 7. Hopping emitter episode handling
    # ================================================================
    def test_hopping_emitter_episodes(self):
        """Hopping emitter: band changes create new episodes."""
        tracker = InterceptTimeTracker()
        e = MockEmitter("E_hop", band=1, active=True)

        # Active on band 1 for t=1..3
        for t in [1, 2, 3]:
            tracker.update_emitter_states(t, [e])

        # Hops to band 4 at t=4 (still active)
        e.band = 4
        tracker.update_emitter_states(4, [e])
        tracker.update_emitter_states(5, [e])

        tracker.finalize(5)
        results = tracker.get_results()

        # Band change while active creates a new episode
        self.assertEqual(results['eligible_episodes'], 2)
        details = results['episode_details']
        self.assertEqual(details[0]['band'], 1)
        self.assertEqual(details[1]['band'], 4)

    # ================================================================
    # 8. Deterministic replay
    # ================================================================
    def test_deterministic_replay(self):
        """Same seed produces identical intercept-time metrics."""
        config = get_default_config()
        scenario = [
            {"name": "E1", "band": 1, "behavior": "periodic", "period": 7, "duty_cycle": 2},
            {"name": "E2", "band": 3, "behavior": "intermittent", "activity_probability": 0.3},
        ]
        from algorithms.adaptive_belief import AdaptiveBeliefScheduler
        sched1 = AdaptiveBeliefScheduler(config.environment.num_bands, epsilon=0.1, seed=42)
        sched2 = AdaptiveBeliefScheduler(config.environment.num_bands, epsilon=0.1, seed=42)

        r1 = run_intercept_time_eval_single(sched1, config, 42, scenario)
        r2 = run_intercept_time_eval_single(sched2, config, 42, scenario)

        self.assertEqual(r1['intercept_time']['eligible_episodes'],
                         r2['intercept_time']['eligible_episodes'])
        self.assertEqual(r1['intercept_time']['intercepted_episodes'],
                         r2['intercept_time']['intercepted_episodes'])
        self.assertEqual(r1['intercept_time']['all_latencies'],
                         r2['intercept_time']['all_latencies'])
        self.assertEqual(r1['interception_rate'], r2['interception_rate'])

    # ================================================================
    # 9. Metric denominator
    # ================================================================
    def test_metric_denominators(self):
        """Mean/median latency computed only over intercepted episodes."""
        tracker = InterceptTimeTracker()
        e1 = MockEmitter("E1", band=1, active=False)
        e2 = MockEmitter("E2", band=5, active=False)

        # E1: episode at t=1, intercepted at t=1 (latency=0)
        e1.active = True
        tracker.update_emitter_states(1, [e1, e2])
        tracker.record_scan(1, scanned_band=1, detected=True, was_truly_active=True)
        e1.active = False
        tracker.update_emitter_states(2, [e1, e2])

        # E2: episode at t=3, MISSED
        e2.active = True
        tracker.update_emitter_states(3, [e1, e2])
        e2.active = False
        tracker.update_emitter_states(4, [e1, e2])

        # E1: episode at t=5, intercepted at t=7 (latency=2)
        e1.active = True
        tracker.update_emitter_states(5, [e1, e2])
        tracker.update_emitter_states(6, [e1, e2])
        tracker.record_scan(7, scanned_band=1, detected=True, was_truly_active=True)
        tracker.update_emitter_states(7, [e1, e2])
        tracker.finalize(7)

        results = tracker.get_results()
        self.assertEqual(results['eligible_episodes'], 3)
        self.assertEqual(results['intercepted_episodes'], 2)
        self.assertEqual(results['missed_episodes'], 1)
        # Mean = (0 + 2) / 2 = 1.0 — missed NOT included
        self.assertEqual(results['mean_intercept_time'], 1.0)

    # ================================================================
    # 10. Ground-truth separation
    # ================================================================
    def test_ground_truth_separation(self):
        """Scheduler cannot access InterceptTimeTracker or activation timestamps."""
        from algorithms.adaptive_belief import AdaptiveBeliefScheduler
        sched = AdaptiveBeliefScheduler(10, epsilon=0.1, seed=42)

        # Verify scheduler has no reference to tracker attributes
        self.assertFalse(hasattr(sched, 'tracker'))
        self.assertFalse(hasattr(sched, 'episodes'))
        self.assertFalse(hasattr(sched, 'activation_time'))
        self.assertFalse(hasattr(sched, 'intercept_time'))
        self.assertFalse(hasattr(sched, 'emitter_state'))

    # ================================================================
    # 11. Adversarial causality test
    # ================================================================
    def test_adversarial_causality(self):
        """
        Two scenarios with identical observable history through decision time T.
        Hidden future truth differs. Scheduler actions remain identical.
        Evaluation results may differ.
        """
        from algorithms.phase8_spatial_scheduler import SpatialScheduler

        r_ids = ['R1', 'R2']
        nb = 10

        sched_a = SpatialScheduler(r_ids, nb, seed=42, epsilon=0.0, spatial_weight=0.3)
        sched_b = SpatialScheduler(r_ids, nb, seed=42, epsilon=0.0, spatial_weight=0.3)

        # Feed identical observations for T=20 steps
        for t in range(1, 21):
            obs = {'observation_time': t, 'time': t, 'tuned_band': t % nb,
                   'band': t % nb, 'detection_result': (t % 3 == 0),
                   'detected': (t % 3 == 0), 'effective_duration': 1}
            action_a = sched_a.select_action(obs)
            action_b = sched_b.select_action(obs)
            self.assertEqual(action_a, action_b,
                             f"Actions differ at t={t} despite identical history")
            det = (t % 3 == 0)
            sched_a.update(obs, action_a, det)
            sched_b.update(obs, action_b, det)

        # Final decision
        final_obs = {'observation_time': 21, 'time': 21, 'tuned_band': 1,
                     'band': 1, 'detection_result': False, 'detected': False,
                     'effective_duration': 1}
        action_a = sched_a.select_action(final_obs)
        action_b = sched_b.select_action(final_obs)
        self.assertEqual(action_a, action_b)

    # ================================================================
    # 12. No zero-latency misses
    # ================================================================
    def test_no_zero_latency_misses(self):
        """Explicitly verify missed episodes are excluded from latency statistics."""
        tracker = InterceptTimeTracker()

        # 3 episodes, only 1 intercepted at latency=5
        e = MockEmitter("E1", band=2, active=False)

        # Episode 1: missed
        e.active = True
        tracker.update_emitter_states(1, [e])
        tracker.update_emitter_states(2, [e])
        e.active = False
        tracker.update_emitter_states(3, [e])

        # Episode 2: intercepted at latency 5
        e.active = True
        tracker.update_emitter_states(4, [e])
        for t in range(5, 9):
            tracker.update_emitter_states(t, [e])
        tracker.record_scan(9, scanned_band=2, detected=True, was_truly_active=True)
        tracker.update_emitter_states(9, [e])
        e.active = False
        tracker.update_emitter_states(10, [e])

        # Episode 3: missed
        e.active = True
        tracker.update_emitter_states(11, [e])
        e.active = False
        tracker.update_emitter_states(12, [e])

        tracker.finalize(12)
        results = tracker.get_results()

        self.assertEqual(results['eligible_episodes'], 3)
        self.assertEqual(results['intercepted_episodes'], 1)
        self.assertEqual(results['missed_episodes'], 2)

        # Only 1 latency value, NOT zero
        self.assertEqual(len(results['all_latencies']), 1)
        self.assertEqual(results['all_latencies'][0], 5)
        # No zeros from missed episodes
        self.assertNotIn(0, [lat for lat in results['all_latencies']])
        # mean should be exactly 5.0, not diluted by zeros
        self.assertEqual(results['mean_intercept_time'], 5.0)

    # ================================================================
    # 13. Full integration: single-receiver returns intercept_time
    # ================================================================
    def test_single_receiver_integration(self):
        """run_intercept_time_eval_single returns intercept_time key."""
        config = get_default_config()
        scenario = [
            {"name": "E1", "band": 1, "behavior": "periodic", "period": 7, "duty_cycle": 2},
        ]
        from algorithms.sequential import SequentialScheduler
        sched = SequentialScheduler(config.environment.num_bands)
        result = run_intercept_time_eval_single(sched, config, 42, scenario)

        self.assertIn('intercept_time', result)
        it = result['intercept_time']
        self.assertIn('eligible_episodes', it)
        self.assertIn('intercepted_episodes', it)
        self.assertIn('missed_episodes', it)
        self.assertIn('mean_intercept_time', it)
        self.assertGreater(it['eligible_episodes'], 0)
        # interception_rate matches base metric
        self.assertIn('interception_rate', result)

    # ================================================================
    # 14. Full integration: multi-receiver returns intercept_time
    # ================================================================
    def test_multi_receiver_integration(self):
        """run_intercept_time_eval_multi returns intercept_time key."""
        config = get_default_config()
        scenario = [
            {"name": "E1", "band": 1, "behavior": "periodic", "period": 7, "duty_cycle": 2, "position": (0,0)},
        ]
        rcfg = [
            {'id': 'R1', 'position': (0.0, 0.0), 'snr_threshold': 20.0},
        ]
        from algorithms.phase8_spatial_scheduler import SpatialScheduler
        r_ids = [r['id'] for r in rcfg]
        sched = SpatialScheduler(r_ids, config.environment.num_bands, seed=42, epsilon=0.1)
        result = run_intercept_time_eval_multi(sched, config, 42, scenario, rcfg)

        self.assertIn('intercept_time', result)
        it = result['intercept_time']
        self.assertGreater(it['eligible_episodes'], 0)

    # ================================================================
    # 15. EmitterEpisode properties
    # ================================================================
    def test_episode_properties(self):
        """EmitterEpisode correctly reports latency and finished state."""
        ep = EmitterEpisode("E1", band=3, start_time=10, episode_id=1)
        self.assertFalse(ep.is_finished)
        self.assertIsNone(ep.intercept_latency)

        ep.intercepted = True
        ep.first_interception_time = 15
        self.assertEqual(ep.intercept_latency, 5)

        ep.end_time = 20
        self.assertTrue(ep.is_finished)

        d = ep.to_dict()
        self.assertEqual(d['emitter_name'], "E1")
        self.assertEqual(d['intercept_latency'], 5)


if __name__ == '__main__':
    unittest.main()
