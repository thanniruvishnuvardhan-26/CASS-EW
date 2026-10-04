import unittest
import numpy as np

from algorithms.phase5_temporal_profile import BandTemporalProfile, MultiBandTemporalProfileScheduler


class TestMultiBandTemporalProfile(unittest.TestCase):

    def setUp(self):
        self.profile = BandTemporalProfile(
            history_size=10,
            minimum_detections=3,
            period_window_size=5,
            temporal_tolerance=2,
            freshness_decay_rate=0.01
        )

    def test_a_independent_band_histories(self):
        scheduler = MultiBandTemporalProfileScheduler(num_bands=2, minimum_detections=3, seed=42)
        scheduler.update(10, 0, True)
        scheduler.update(20, 0, True)
        scheduler.update(30, 0, True)
        
        # Band 0 should have period 10
        p0, _, _ = scheduler.profiles[0].get_period_stats()
        self.assertEqual(p0, 10.0)
        
        # Band 1 should have insufficient history
        p1, _, _ = scheduler.profiles[1].get_period_stats()
        self.assertIsNone(p1)

    def test_b_perfect_periodic_pattern(self):
        for t in [10, 20, 30, 40]:
            self.profile.update(t, True)
        
        p, num_ints, mad = self.profile.get_period_stats()
        self.assertEqual(p, 10.0)
        self.assertEqual(mad, 0.0)
        self.assertEqual(num_ints, 3)

    def test_c_jittered_periodic_pattern(self):
        for t in [10, 21, 30, 42, 50, 63]:
            self.profile.update(t, True)
            
        p, num_ints, mad = self.profile.get_period_stats()
        self.assertIsNotNone(p)
        self.assertTrue(9.0 <= p <= 12.0)
        self.assertGreater(mad, 0.0)

    def test_d_harmonic_missed_detections(self):
        for t in [10, 30, 50, 70]:
            self.profile.update(t, True)
            
        p, num_ints, mad = self.profile.get_period_stats()
        self.assertEqual(p, 20.0)
        self.assertEqual(mad, 0.0)

    def test_e_insufficient_evidence(self):
        self.profile.update(10, True)
        self.profile.update(20, True)
        
        p, _, _ = self.profile.get_period_stats()
        self.assertIsNone(p)
        self.assertEqual(self.profile.temporal_reliability(), 0.0)
        
        # Temporal prediction score should be 0.5 when unavailable
        self.assertEqual(self.profile.temporal_prediction_score(30), 0.5)

    def test_f_freshness(self):
        for t in [10, 20, 30, 40]:
            self.profile.update(t, True)
            
        f1 = self.profile.temporal_freshness(40)
        f2 = self.profile.temporal_freshness(50)
        f3 = self.profile.temporal_freshness(100)
        
        self.assertEqual(f1, 1.0)
        self.assertTrue(f2 < f1)
        self.assertTrue(f3 < f2)
        
        f4 = self.profile.temporal_freshness(200)
        self.assertEqual(f4, 0.0) # bounded at 0

    def test_g_reliability(self):
        # Perfect pattern
        for t in [10, 20, 30, 40, 50, 60]:
            self.profile.update(t, True)
            
        rel_high = self.profile.temporal_reliability()
        
        self.profile.reset()
        # Jittered pattern
        for t in [10, 23, 31, 48, 55, 69]:
            self.profile.update(t, True)
            
        rel_low = self.profile.temporal_reliability()
        
        self.assertTrue(rel_high > rel_low)
        self.assertGreater(rel_high, 0.0)

    def test_h_prediction(self):
        for t in [10, 20, 30, 40]:
            self.profile.update(t, True)
            
        # Expected at 50
        score_at_50 = self.profile.temporal_prediction_score(50)
        self.assertEqual(score_at_50, 1.0)
        
        # Linear decay away from 50 (tolerance is 2)
        score_at_53 = self.profile.temporal_prediction_score(53)
        self.assertTrue(0.0 <= score_at_53 < 1.0)
        
        score_at_55 = self.profile.temporal_prediction_score(55)
        self.assertEqual(score_at_55, 0.0)

    def test_i_priority(self):
        scheduler = MultiBandTemporalProfileScheduler(
            num_bands=2, 
            belief_weight=0.5, 
            temporal_weight=0.5, 
            epsilon=0.0,
            minimum_detections=3,
            temporal_tolerance=2
        )
        
        # Band 0: High belief, weak/no temporal
        scheduler.beliefs[0] = 0.9
        
        # Band 1: Moderate belief, strong reliable fresh temporal
        for t in [10, 20, 30, 40]:
            scheduler.update({'observation_time': t}, 1, True)
        scheduler.beliefs[1] = 0.4
            
        # At time 50, Band 1 is expected
        action = scheduler.select_action({'observation_time': 50})
        # Band 1 Priority: 0.5*0.4 + 0.5*(1.0 * rel * fresh) = 0.2 + 0.5*rel*fresh.
        # rel for 3 intervals, period_window_size 5 is 3/5 = 0.6.
        # fresh = 1.0. 0.2 + 0.5*1.0*0.6*1.0 = 0.5
        # Band 0 Priority: 0.5*0.9 + 0.5*(0.5 * 0 * 0) = 0.45
        # So Band 1 should win
        self.assertEqual(action, 1)
        
        # At time 55, Band 1 is NOT expected (t_score=0)
        # Band 1 Priority: 0.5*0.4 + 0.5*(0 * rel * fresh) = 0.2
        # Band 0 Priority: 0.45
        # So Band 0 should win
        action_2 = scheduler.select_action({'observation_time': 55})
        self.assertEqual(action_2, 0)

    def test_j_starvation(self):
        scheduler = MultiBandTemporalProfileScheduler(num_bands=3, epsilon=0.5, seed=10)
        # Bounded exploration is enabled
        selections = []
        for i in range(100):
            action = scheduler.select_action({'observation_time': i})
            scheduler.update(i, action, action == 0) # Only 0 gives detection
            selections.append(action)
            
        self.assertIn(1, selections)
        self.assertIn(2, selections)

    def test_k_reset(self):
        scheduler = MultiBandTemporalProfileScheduler(num_bands=2)
        scheduler.update(10, 0, True)
        scheduler.update(20, 0, True)
        
        self.assertEqual(len(scheduler.profiles[0].detections), 2)
        
        scheduler.reset()
        self.assertEqual(len(scheduler.profiles[0].detections), 0)

    def test_l_adversarial_leakage(self):
        """
        Two environments with identical observable history but different ground truth.
        """
        # We simulate the exact same observations being fed to the scheduler.
        sched1 = MultiBandTemporalProfileScheduler(num_bands=2, epsilon=0.1, seed=42)
        sched2 = MultiBandTemporalProfileScheduler(num_bands=2, epsilon=0.1, seed=42)
        
        observations = [
            {'observation_time': 0},
            {'observation_time': 5},
            {'observation_time': 10}
        ]
        
        actions1 = []
        for obs in observations:
            a = sched1.select_action(obs)
            actions1.append(a)
            sched1.update(obs, a, True)
            
        # Env 2 has different ground truth (but observation dict only has time)
        actions2 = []
        for obs in observations:
            a = sched2.select_action(obs)
            actions2.append(a)
            sched2.update(obs, a, True)
            
        self.assertEqual(actions1, actions2)

    def test_m_end_to_end_adversarial_leakage(self):
        from simulator.environment import RFEnvironment, Emitter
        from simulator.receiver import VirtualReceiver
        
        # Env 1: Emitter is always on
        env1 = RFEnvironment(num_bands=2, seed=42)
        env1.add_emitter(Emitter(name="E1", band=0, behavior="periodic", activity_probability=1.0, seed=42))
        
        # Env 2: Emitter is never on
        env2 = RFEnvironment(num_bands=2, seed=42)
        env2.add_emitter(Emitter(name="E2", band=0, behavior="periodic", activity_probability=0.0, seed=42))
        
        sched1 = MultiBandTemporalProfileScheduler(num_bands=2, epsilon=0.1, seed=100)
        sched2 = MultiBandTemporalProfileScheduler(num_bands=2, epsilon=0.1, seed=100)
        
        receiver1 = VirtualReceiver(num_bands=2, detection_probability=1.0, false_alarm_probability=0.0, switching_time=1, seed=42)
        receiver2 = VirtualReceiver(num_bands=2, detection_probability=1.0, false_alarm_probability=0.0, switching_time=1, seed=42)
        
        actions1 = []
        actions2 = []
        
        # Force the receiver2 to have same observation outcome as receiver1 by mocking the return
        # But we don't even need to mock if we just force the `detected` result to be identical.
        # Actually, if env1 and env2 have different ground truth, receiver1 and receiver2 will produce different `detected` results.
        # To make "receiver-observable histories presented to the scheduler are identical", we must intercept the scan result and force it to be identical,
        # OR we just test that the scheduler does not break if we feed it the same observation history.
        # Wait, if we use receiver.scan(env, action, dwell=1), receiver1 gets True, receiver2 gets False.
        # The prompt says: "two RF environments have different hidden ground truth / receiver-observable histories presented to the scheduler are identical"
        # We can just manually construct identical `detected` results for both despite the different environments.
        
        for t in range(50):
            env1.step()
            env2.step()
            
            obs1 = receiver1.scan_history[-1] if receiver1.scan_history else None
            obs2 = receiver2.scan_history[-1] if receiver2.scan_history else None
            
            a1 = sched1.select_action(obs1)
            a2 = sched2.select_action(obs2)
            
            actions1.append(a1)
            actions2.append(a2)
            
            # They scan their respective environments
            d1 = receiver1.scan(env1, a1, dwell_time=1)
            d2 = receiver2.scan(env2, a2, dwell_time=1)
            
            # FORCE identical observation history (e.g. both see a detection if t % 5 == 0)
            forced_detection = (t % 5 == 0)
            
            # The observation dictionaries must be identical in terms of timing
            # The scheduler takes obs and detected
            sched1.update(receiver1.scan_history[-1], a1, forced_detection)
            sched2.update(receiver2.scan_history[-1], a2, forced_detection)
            
        self.assertEqual(actions1, actions2)

if __name__ == '__main__':
    unittest.main()
