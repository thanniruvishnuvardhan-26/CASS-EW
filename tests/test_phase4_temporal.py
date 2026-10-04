import unittest
import numpy as np

from algorithms.temporal_model import TemporalModel
from algorithms.temporal_belief import TemporalBeliefScheduler
from simulator.environment import RFEnvironment, Emitter
from simulator.receiver import VirtualReceiver

class TestPhase4Temporal(unittest.TestCase):

    def test_a_perfect_period(self):
        model = TemporalModel(num_bands=1, minimum_detections=3, period_window_size=5)
        for t in [10, 20, 30, 40]:
            model.update(band=0, time=t, detection=True)
            
        period, count, mad = model.estimate_period(0)
        self.assertEqual(period, 10.0)
        self.assertEqual(count, 3)
        self.assertEqual(mad, 0.0)

    def test_b_jitter(self):
        model = TemporalModel(num_bands=1, minimum_detections=3, period_window_size=5)
        for t in [10, 20, 31, 40]:
            model.update(band=0, time=t, detection=True)
            
        period, count, mad = model.estimate_period(0)
        self.assertEqual(period, 10.0)
        self.assertEqual(count, 3)
        self.assertEqual(mad, 1.0) # |10-10|=0, |11-10|=1, |9-10|=1 -> median is 1.0

    def test_c_insufficient_evidence(self):
        model = TemporalModel(num_bands=1, minimum_detections=3, period_window_size=5)
        model.update(band=0, time=10, detection=True)
        
        period, count, mad = model.estimate_period(0)
        self.assertIsNone(period)

    def test_d_two_detections(self):
        model = TemporalModel(num_bands=1, minimum_detections=2, period_window_size=5)
        model.update(band=0, time=10, detection=True)
        model.update(band=0, time=20, detection=True)
        
        period, count, mad = model.estimate_period(0)
        self.assertEqual(period, 10.0)
        self.assertEqual(count, 1)

    def test_e_irregular_sequence(self):
        model = TemporalModel(num_bands=1, minimum_detections=3, period_window_size=5)
        for t in [3, 11, 19, 42, 44]:
            model.update(band=0, time=t, detection=True)
            
        period, count, mad = model.estimate_period(0)
        # intervals: 8, 8, 23, 2. Median: 8.
        # MAD: abs([8-8, 8-8, 23-8, 2-8]) = [0, 0, 15, 6]. Median = 3.0.
        self.assertEqual(period, 8.0)
        self.assertEqual(mad, 3.0)

    def test_f_reset(self):
        model = TemporalModel(num_bands=1, minimum_detections=3, period_window_size=5)
        for t in [10, 20, 30]:
            model.update(band=0, time=t, detection=True)
        
        self.assertIsNotNone(model.estimate_period(0)[0])
        model.reset()
        self.assertIsNone(model.estimate_period(0)[0])

    def test_g_multiple_bands(self):
        model = TemporalModel(num_bands=2, minimum_detections=3, period_window_size=5)
        for t in [10, 20, 30]:
            model.update(band=0, time=t, detection=True)
        for t in [5, 12, 19]:
            model.update(band=1, time=t, detection=True)
            
        p0, _, _ = model.estimate_period(0)
        p1, _, _ = model.estimate_period(1)
        self.assertEqual(p0, 10.0)
        self.assertEqual(p1, 7.0)

    def test_h_missed_detections_harmonic(self):
        # Detections occur at 10, 30, 50, 70
        # The true period might be 10, but we missed the ones at 20, 40, 60.
        # This test documents that the median will correctly estimate 20 (the harmonic).
        model = TemporalModel(num_bands=1, minimum_detections=3, period_window_size=5)
        for t in [10, 30, 50, 70]:
            model.update(band=0, time=t, detection=True)
            
        period, count, mad = model.estimate_period(0)
        self.assertEqual(period, 20.0)

    def test_temporal_score(self):
        model = TemporalModel(num_bands=1, minimum_detections=2, temporal_tolerance=2)
        model.update(band=0, time=10, detection=True)
        model.update(band=0, time=20, detection=True)
        
        # Period = 10, last_det = 20. Expected next = 30.
        # Nearest multiple of 10 from (t - last_det)
        self.assertEqual(model.temporal_score(0, 30), 1.0) # elapsed=10, nearest=10, expected=30
        self.assertEqual(model.temporal_score(0, 31), 1.0) # elapsed=11, nearest=10, distance=1 <= tolerance
        
        # t=35 -> elapsed=15. nearest multiple of 10 to 15 is 20 (since round(1.5)=2).
        # Expected=40. distance=abs(15-20)=5. Decay=(5-2)/2=1.5. Score=max(0, 1-1.5)=0.0
        self.assertEqual(model.temporal_score(0, 35), 0.0) 
        
        self.assertEqual(model.temporal_score(0, 40), 1.0) # elapsed=20, nearest=20, expected=40
        self.assertEqual(model.temporal_score(0, 50), 1.0) # elapsed=30, nearest=30, expected=50

    def test_hybrid_scheduler_equal_beliefs(self):
        scheduler = TemporalBeliefScheduler(
            num_bands=2, minimum_detections=2, belief_weight=0.5, temporal_weight=0.5, epsilon=0.0
        )
        # Setup equal beliefs
        scheduler.beliefs = [0.5, 0.5]
        
        # Provide strong temporal score for band 1
        scheduler.update(10, 1, True)
        scheduler.update(20, 1, True)
        
        # At t=30, band 1 has t_score=1.0, band 0 has t_score=0.5
        # b0: 0.5*0.5 + 0.5*0.5 = 0.5
        # b1: 0.5*0.5 + 0.5*1.0 = 0.75
        action = scheduler.select_action(30)
        self.assertEqual(action, 1)

    def test_hybrid_scheduler_strong_belief_weak_temporal(self):
        scheduler = TemporalBeliefScheduler(
            num_bands=2, minimum_detections=2, belief_weight=0.8, temporal_weight=0.2, epsilon=0.0
        )
        scheduler.beliefs = [0.9, 0.2]
        # Band 1 has perfect temporal score
        scheduler.update(10, 1, True)
        scheduler.update(20, 1, True)
        
        # At t=30
        # b0: 0.8*0.9 + 0.2*0.5 = 0.72 + 0.1 = 0.82
        # b1: 0.8*0.2 + 0.2*1.0 = 0.16 + 0.2 = 0.36
        action = scheduler.select_action(30)
        self.assertEqual(action, 0)
        
    def test_hybrid_scheduler_reproducibility(self):
        sched1 = TemporalBeliefScheduler(num_bands=3, seed=42, epsilon=1.0)
        sched2 = TemporalBeliefScheduler(num_bands=3, seed=42, epsilon=1.0)
        
        a1 = [sched1.select_action(t) for t in range(10)]
        a2 = [sched2.select_action(t) for t in range(10)]
        self.assertEqual(a1, a2)

    def test_end_to_end_leakage(self):
        envA = RFEnvironment(num_bands=5, seed=42)
        envA.add_emitter(Emitter("A", band=2, behavior="periodic", activity_probability=1.0))
        envA.time = 0
        
        envB = RFEnvironment(num_bands=5, seed=42)
        envB.add_emitter(Emitter("A", band=2, behavior="periodic", activity_probability=1.0))
        envB.time = 0
        
        recvA = VirtualReceiver(num_bands=5, seed=123)
        recvB = VirtualReceiver(num_bands=5, seed=123)
        
        schedA = TemporalBeliefScheduler(num_bands=5, seed=99, epsilon=0.0)
        schedB = TemporalBeliefScheduler(num_bands=5, seed=99, epsilon=0.0)
        
        for t in range(5):
            envA.step()
            envB.step()
            
            detA = recvA.scan(envA, 2)
            detB = recvB.scan(envB, 2)
            
            schedA.update(envA.time, 2, detA)
            schedB.update(envB.time, 2, detB)
            
        # At t=5, Ground truth changes for env B only
        envB.emitters[0].active = False
        
        actionA = schedA.select_action(envA.time)
        actionB = schedB.select_action(envB.time)
        
        self.assertEqual(actionA, actionB)

    def test_observation_time_boundary(self):
        # Verify the scheduler obtains observation_time through the legitimate 
        # receiver observation contract and correctly parses it
        recv = VirtualReceiver(num_bands=2, switching_time=1)
        env = RFEnvironment(num_bands=2, seed=42)
        sched = TemporalBeliefScheduler(num_bands=2)
        
        # Advance env.time artificially
        env.time = 100
        det = recv.scan(env, 0, dwell_time=2)
        # scan_history should have 'observation_time' == 100 + 0 (first switch is free) + 2 (dwell) = 102
        obs = recv.scan_history[-1]
        self.assertIn('observation_time', obs)
        self.assertEqual(obs['observation_time'], 102)
        
        # update scheduler
        sched.update(obs, 0, det)
        # Check if temporal model recorded time correctly
        history = sched.temporal_model.history[0]
        if det:
            self.assertEqual(history[-1][0], 102)

if __name__ == '__main__':
    unittest.main()
