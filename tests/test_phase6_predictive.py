import unittest
import numpy as np

from algorithms.phase6_predictive_scheduler import BandPredictiveProfile, PredictiveScheduler
from simulator.environment import RFEnvironment
from simulator.receiver import VirtualReceiver

class TestBandPredictiveProfile(unittest.TestCase):
    def test_a_perfect_periodic_detections(self):
        prof = BandPredictiveProfile()
        # 3 detections, period = 10
        prof.update(10, True)
        prof.update(20, True)
        prof.update(30, True)
        
        self.assertIsNotNone(prof.next_predicted_time)
        self.assertEqual(prof.next_predicted_time, 40)
        self.assertTrue(prof.prediction_confidence > 0)
        
    def test_b_prediction_with_jitter(self):
        prof = BandPredictiveProfile(temporal_tolerance=2)
        prof.update(10, True)
        prof.update(21, True)
        prof.update(29, True)
        
        p, num, mad = prof.get_period_stats()
        self.assertIsNotNone(p)
        self.assertIsNotNone(prof.next_predicted_time)
        
    def test_c_high_mad_confidence_reduction(self):
        prof_stable = BandPredictiveProfile()
        prof_stable.update(10, True)
        prof_stable.update(20, True)
        prof_stable.update(30, True)
        conf_stable = prof_stable.prediction_confidence
        
        prof_jitter = BandPredictiveProfile()
        prof_jitter.update(10, True)
        prof_jitter.update(25, True)
        prof_jitter.update(30, True)
        conf_jitter = prof_jitter.prediction_confidence
        
        self.assertTrue(conf_stable > conf_jitter)
        
    def test_d_insufficient_history(self):
        prof = BandPredictiveProfile()
        prof.update(10, True)
        prof.update(20, True)
        self.assertIsNone(prof.next_predicted_time)
        self.assertEqual(prof.prediction_confidence, 0.0)
        
    def test_e_stale_prediction(self):
        prof = BandPredictiveProfile()
        prof.update(10, True)
        prof.update(20, True)
        prof.update(30, True)
        
        conf_fresh = prof.prediction_confidence
        
        # Advance time significantly without detection
        prof.update(100, False)
        conf_stale = prof.prediction_confidence
        
        self.assertTrue(conf_fresh > conf_stale)
        
    def test_f_harmonic_period(self):
        prof = BandPredictiveProfile()
        prof.update(10, True)
        prof.update(30, True) # Missed 20
        prof.update(50, True) # Missed 40
        self.assertEqual(prof.next_predicted_time, 70)
        
    def test_g_h_i_prediction_evaluation(self):
        prof = BandPredictiveProfile(temporal_tolerance=2)
        prof.update(10, True)
        prof.update(20, True)
        prof.update(30, True)
        
        # Next predicted is 40.
        self.assertEqual(prof.prediction_count, 0)
        
        # A hit at 40
        prof.update(40, True)
        self.assertEqual(prof.prediction_count, 1)
        self.assertEqual(prof.prediction_hits, 1)
        self.assertEqual(prof.prediction_absolute_errors[-1], 0)
        
        # Next predicted is 50.
        # A hit at 51
        prof.update(51, True)
        self.assertEqual(prof.prediction_count, 2)
        self.assertEqual(prof.prediction_hits, 2)
        self.assertEqual(prof.prediction_absolute_errors[-1], 1)
        
        # Next predicted is 61.
        # A miss at 65 (tolerance is 2)
        prof.update(65, True)
        self.assertEqual(prof.prediction_count, 3)
        self.assertEqual(prof.prediction_hits, 2)
        self.assertEqual(prof.prediction_misses, 1)
        self.assertEqual(prof.prediction_absolute_errors[-1], 4)

    def test_j_prediction_confidence_bounds(self):
        prof = BandPredictiveProfile()
        prof.update(10, True)
        prof.update(20, True)
        prof.update(30, True)
        self.assertTrue(0.0 <= prof.prediction_confidence <= 1.0)
        
    def test_k_reset_clears_state(self):
        prof = BandPredictiveProfile()
        prof.update(10, True)
        prof.update(20, True)
        prof.update(30, True)
        prof.reset()
        self.assertIsNone(prof.next_predicted_time)
        self.assertEqual(prof.prediction_count, 0)
        
class TestPredictiveScheduler(unittest.TestCase):
    def test_l_seeded_reproducibility(self):
        sched1 = PredictiveScheduler(num_bands=5, seed=42)
        sched2 = PredictiveScheduler(num_bands=5, seed=42)
        
        self.assertEqual(sched1.select_action(0), sched2.select_action(0))
        
    def test_m_n_exploration_predictive_influence(self):
        sched = PredictiveScheduler(num_bands=5, epsilon=0.0) # pure exploitation
        sched.beliefs = [0.1, 0.9, 0.1, 0.1, 0.1]
        
        # A: prediction has no effect -> influence = 0
        action = sched.select_action(0)
        self.assertEqual(action, 1) # belief alone wins
        self.assertEqual(sched.predictive_influenced_selections, 0)
        
        # B: prediction changes selected action -> influence = 1
        prof0 = sched.profiles[0]
        for t in [10, 20, 30, 40, 50, 60]:
            prof0.update(t, True)
        
        # At time 70, Band 0 predictive + temporal score beats Band 1 belief
        # but Band 0 temporal alone does not beat Band 1 belief
        action = sched.select_action(70)
        self.assertEqual(action, 0)
        self.assertEqual(sched.predictive_influenced_selections, 1)
        
        # C: exploration -> influence is not counted
        sched.epsilon = 1.0 # Force exploration
        action = sched.select_action(71)
        self.assertEqual(sched.exploration_selections, 1)
        self.assertEqual(sched.predictive_influenced_selections, 1) # unchanged
        
    def test_o_observation_boundary_timing(self):
        sched = PredictiveScheduler(num_bands=5)
        # Pass a dict instead of relying on env.time
        action = sched.select_action({"observation_time": 42})
        self.assertIsNotNone(action)
        
    def test_p_direct_env_time_leakage(self):
        sched = PredictiveScheduler(num_bands=5)
        
        # Should not crash if observation is an int
        action = sched.select_action(100)
        self.assertIsNotNone(action)
        
    def test_q_adversarial_leakage(self):
        # Two environments with identical history up to T=30
        
        env1 = RFEnvironment(num_bands=5)
        from simulator.environment import Emitter
        env1.add_emitter(Emitter(name="E1", band=0, behavior="periodic", activity_probability=1.0, seed=42))
        
        env2 = RFEnvironment(num_bands=5)
        env2.add_emitter(Emitter(name="E2", band=0, behavior="periodic", activity_probability=0.0, seed=42)) # Different truth
        
        rx1 = VirtualReceiver(num_bands=5, switching_time=0)
        rx2 = VirtualReceiver(num_bands=5, switching_time=0)
        
        sched1 = PredictiveScheduler(num_bands=5, seed=42)
        sched2 = PredictiveScheduler(num_bands=5, seed=42)
        
        for t in range(31):
            obs = {"observation_time": t}
            # Feed exactly the same observation history to both schedulers
            forced_detection = (t % 10 == 0) # Just simulate some identical history
            sched1.update(obs, 0, forced_detection)
            sched2.update(obs, 0, forced_detection)
            
            env1.step()
            env2.step()
            
        action1 = sched1.select_action({"observation_time": 31})
        action2 = sched2.select_action({"observation_time": 31})
        
        self.assertEqual(action1, action2)

if __name__ == '__main__':
    unittest.main()
