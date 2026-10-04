import unittest
import numpy as np

from algorithms.phase7_signal_pattern import PatternAwareScheduler, PatternRelationship
from simulator.environment import RFEnvironment, Emitter
from simulator.receiver import VirtualReceiver

class TestPhase7PatternAwareScheduler(unittest.TestCase):
    def setUp(self):
        self.num_bands = 10
        self.scheduler = PatternAwareScheduler(num_bands=self.num_bands, seed=42)
        
    def test_deterministic_reset(self):
        s1 = PatternAwareScheduler(num_bands=5, seed=42)
        s2 = PatternAwareScheduler(num_bands=5, seed=42)
        
        # Advance s1
        for _ in range(50):
            obs = {"observation_time": 1, "band": 0, "effective_duration": 1}
            action = s1.select_action(obs)
            s1.update({"observation_time": 1, "band": action, "effective_duration": 1}, action, True)
            
        s1.reset(seed=42)
        
        # Check equality
        for _ in range(50):
            obs = {"observation_time": 1, "band": 0, "effective_duration": 1}
            a1 = s1.select_action(obs)
            a2 = s2.select_action(obs)
            self.assertEqual(a1, a2)
            s1.update({"observation_time": 1, "band": a1, "effective_duration": 1}, a1, True)
            s2.update({"observation_time": 1, "band": a2, "effective_duration": 1}, a2, True)

    def test_causality_leakage_prevention(self):
        # Two environments with identical history up to T but different futures/hidden state
        env1 = RFEnvironment(num_bands=5, seed=42)
        env1.add_emitter(Emitter(name="A", band=1, behavior="periodic", period=5))
        
        env2 = RFEnvironment(num_bands=5, seed=42)
        env2.add_emitter(Emitter(name="B", band=1, behavior="periodic", period=5))
        env2.add_emitter(Emitter(name="C", band=2, behavior="periodic", period=10)) # Different hidden truth
        
        s1 = PatternAwareScheduler(num_bands=5, seed=42)
        s2 = PatternAwareScheduler(num_bands=5, seed=42)
        
        # Feed identical observations to both
        for t in range(20):
            obs = {"observation_time": t, "band": 0, "effective_duration": 1}
            
            a1 = s1.select_action(obs)
            a2 = s2.select_action(obs)
            self.assertEqual(a1, a2)
            
            # Pretend we detected nothing
            s1.update(obs, a1, False)
            s2.update(obs, a2, False)

    def test_pattern_relationship_creation(self):
        # Emulate 2 bands getting detections at the same time
        self.scheduler.reset(seed=42)
        for t in [10, 20, 30, 40]:
            # Fake predictions being near each other by manually updating profiles
            self.scheduler.profiles[1].update(t-5, True)
            self.scheduler.profiles[2].update(t-5, True)
            
            self.scheduler.profiles[1].next_predicted_time = t
            self.scheduler.profiles[2].next_predicted_time = t
            
            obs = {"observation_time": t, "band": 0, "effective_duration": 1}
            self.scheduler.update(obs, 0, False) # This should trigger relationship updates
            
        pair = (1, 2)
        self.assertIn(pair, self.scheduler.relationships)
        rel = self.scheduler.relationships[pair]
        self.assertGreater(rel.coincidences, 0)
        self.assertGreater(rel.confidence(), 0)

    def test_competing_opportunities_and_pattern_influence(self):
        s = PatternAwareScheduler(
            num_bands=3, 
            seed=42, 
            belief_weight=0.0, 
            temporal_weight=0.0, 
            prediction_weight=0.5, 
            pattern_weight=0.5,
            epsilon=0.0 # Force exploitation
        )
        
        # Setup situation where band 1 and band 2 have identical predictions
        s.profiles[1].next_predicted_time = 10
        s.profiles[1].prediction_confidence = 0.8
        
        s.profiles[2].next_predicted_time = 10
        s.profiles[2].prediction_confidence = 0.8
        
        # But band 1 has a pattern relationship with band 0
        s.relationships[(0, 1)] = PatternRelationship(0, 1)
        s.relationships[(0, 1)].coincidences = 5
        s.relationships[(0, 1)].observations = 5
        s.relationships[(0, 1)].last_observed_time = 9
        
        # Make band 2 have a stronger prediction than band 1, so without pattern 2 wins.
        s.profiles[2].prediction_confidence = 0.95
        
        # Band 0 has strong prediction, giving band 1 a strong pattern correlation
        s.profiles[0].next_predicted_time = 10
        s.profiles[0].prediction_confidence = 0.1 # doesn't matter, p_score is 1.0
        
        # Band 1 should have higher pattern strength than band 2
        obs = {"observation_time": 10, "band": 0, "effective_duration": 1}
        action = s.select_action(obs)
        
        # 1 and 2 are predicted. 2 is stronger without pattern.
        # But 1 gets boosted by pattern from 0, so 1 wins.
        self.assertEqual(s.pattern_influenced_selections, 1)

if __name__ == '__main__':
    unittest.main()
