import unittest
import numpy as np

from algorithms.adaptive_belief import AdaptiveBeliefScheduler
from algorithms.random_scheduler import RandomScheduler
from algorithms.sequential import SequentialScheduler
from simulator.environment import RFEnvironment, Emitter
from simulator.receiver import VirtualReceiver

class TestPhase3Schedulers(unittest.TestCase):

    def test_01_adaptive_initialization(self):
        """Test scheduler initialization and parameter assignment."""
        sched = AdaptiveBeliefScheduler(
            num_bands=12,
            initial_belief=0.4,
            belief_hit_update=0.25,
            belief_miss_update=0.15,
            epsilon=0.2,
            seed=42
        )
        self.assertEqual(sched.num_bands, 12)
        self.assertEqual(sched.initial_belief, 0.4)
        
        state = sched.get_state()
        self.assertEqual(len(state["beliefs"]), 12)
        self.assertTrue(np.allclose(state["beliefs"], 0.4))
        self.assertTrue(np.all(state["observations"] == 0))
        self.assertTrue(np.all(state["detections"] == 0))

    def test_02_adaptive_reset(self):
        """Test reset semantics for AdaptiveBeliefScheduler."""
        sched = AdaptiveBeliefScheduler(num_bands=5, initial_belief=0.5, seed=10)
        # Advance state
        sched.update(None, 0, True)
        sched.update(None, 1, False)
        
        state = sched.get_state()
        self.assertNotEqual(state["beliefs"][0], 0.5)
        self.assertEqual(state["step_counter"], 2)
        
        # Reset
        sched.reset(seed=20)
        state_after = sched.get_state()
        self.assertTrue(np.allclose(state_after["beliefs"], 0.5))
        self.assertEqual(state_after["step_counter"], 0)
        self.assertTrue(np.all(state_after["observations"] == 0))

    def test_03_belief_update_bounds_and_logic(self):
        """Test detection and miss updates, and ensure belief bounds [0, 1]."""
        sched = AdaptiveBeliefScheduler(
            num_bands=3, 
            initial_belief=0.5, 
            belief_hit_update=0.3, 
            belief_miss_update=0.4
        )
        
        # Test hit update
        sched.update(None, 0, True)
        self.assertAlmostEqual(sched.get_state()["beliefs"][0], 0.8)
        
        # Test hit update bound (0.8 + 0.3 = 1.1 -> clamped to 1.0)
        sched.update(None, 0, True)
        self.assertAlmostEqual(sched.get_state()["beliefs"][0], 1.0)
        
        # Test miss update
        sched.update(None, 1, False)
        self.assertAlmostEqual(sched.get_state()["beliefs"][1], 0.1)
        
        # Test miss update bound (0.1 - 0.4 = -0.3 -> clamped to 0.0)
        sched.update(None, 1, False)
        self.assertAlmostEqual(sched.get_state()["beliefs"][1], 0.0)

    def test_04_deterministic_tie_breaking(self):
        """Test deterministic tie breaking (lowest index)."""
        sched = AdaptiveBeliefScheduler(num_bands=5, initial_belief=0.5, epsilon=0.0)
        # All beliefs are equal (0.5), epsilon=0 ensures exploitation
        # Should pick band 0
        self.assertEqual(sched.select_action(), 0)
        
        # Make band 2 and 4 have higher beliefs (e.g. 0.8)
        sched.beliefs[2] = 0.8
        sched.beliefs[4] = 0.8
        
        # Should pick 2, as it's the lowest index among the highest beliefs
        self.assertEqual(sched.select_action(), 2)

    def test_05_epsilon_zero(self):
        """Test that epsilon=0 strictly exploits."""
        sched = AdaptiveBeliefScheduler(num_bands=5, epsilon=0.0)
        sched.beliefs[3] = 0.9
        
        # Try 50 times to ensure no random exploration
        actions = [sched.select_action() for _ in range(50)]
        self.assertTrue(all(a == 3 for a in actions))

    def test_06_epsilon_one(self):
        """Test that epsilon=1 strictly explores."""
        sched = AdaptiveBeliefScheduler(num_bands=5, epsilon=1.0, seed=42)
        sched.beliefs[3] = 0.9
        
        actions = [sched.select_action() for _ in range(100)]
        # Action 3 shouldn't be chosen every time
        self.assertFalse(all(a == 3 for a in actions))
        # It should span valid ranges
        self.assertTrue(all(0 <= a < 5 for a in actions))

    def test_07_seeded_reproducibility(self):
        """Test identical seeds produce identical action sequences."""
        sched1 = AdaptiveBeliefScheduler(num_bands=5, epsilon=0.5, seed=123)
        sched2 = AdaptiveBeliefScheduler(num_bands=5, epsilon=0.5, seed=123)
        
        actions1 = [sched1.select_action() for _ in range(20)]
        actions2 = [sched2.select_action() for _ in range(20)]
        
        self.assertEqual(actions1, actions2)

    def test_08_valid_action_range(self):
        """Test schedulers only produce actions within valid ranges."""
        for num_bands in [2, 15]:
            sched1 = SequentialScheduler(num_bands)
            sched2 = RandomScheduler(num_bands, seed=1)
            sched3 = AdaptiveBeliefScheduler(num_bands, epsilon=0.5, seed=2)
            
            for _ in range(30):
                self.assertTrue(0 <= sched1.select_action() < num_bands)
                self.assertTrue(0 <= sched2.select_action() < num_bands)
                self.assertTrue(0 <= sched3.select_action() < num_bands)

    def test_09_sequential_baseline(self):
        """Test the sequential baseline scheduler."""
        sched = SequentialScheduler(num_bands=3)
        self.assertEqual(sched.select_action(), 0)
        self.assertEqual(sched.select_action(), 1)
        self.assertEqual(sched.select_action(), 2)
        self.assertEqual(sched.select_action(), 0)

    def test_10_random_baseline(self):
        """Test the random baseline scheduler."""
        sched1 = RandomScheduler(num_bands=4, seed=99)
        sched2 = RandomScheduler(num_bands=4, seed=99)
        actions1 = [sched1.select_action() for _ in range(20)]
        actions2 = [sched2.select_action() for _ in range(20)]
        self.assertEqual(actions1, actions2)

    def test_11_end_to_end_leakage(self):
        """
        Verify identical observable histories produce identical scheduler decisions,
        even if future ground truth differs.
        """
        envA = RFEnvironment(num_bands=5)
        envB = RFEnvironment(num_bands=5)
        
        # Scenario A: Emitter A is intermittent, stopping at t=5
        class CustomEmitterA(Emitter):
            def __init__(self, name, band):
                super().__init__(name=name, band=band)
            def step(self):
                # Emitter has a reference to environment but usually it tracks time internally
                # For simplicity, we just use an internal counter
                if not hasattr(self, 't'): self.t = 0
                self.t += 1
                self.active = (self.t <= 5)
                return self.active
        envA.add_emitter(CustomEmitterA("A", 2))
        
        # Scenario B: Emitter B is intermittent, continuing after t=5
        class CustomEmitterB(Emitter):
            def __init__(self, name, band):
                super().__init__(name=name, band=band)
            def step(self):
                if not hasattr(self, 't'): self.t = 0
                self.t += 1
                self.active = True
                return self.active
        envB.add_emitter(CustomEmitterB("B", 2))

        # Both environments have Emitter active continuously from t=0 to t=5
        # So observation up to t=5 should be identical.
        
        recA = VirtualReceiver(num_bands=5, detection_probability=1.0, false_alarm_probability=0.0)
        recB = VirtualReceiver(num_bands=5, detection_probability=1.0, false_alarm_probability=0.0)
        
        schedA = AdaptiveBeliefScheduler(num_bands=5, epsilon=0.2, seed=42)
        schedB = AdaptiveBeliefScheduler(num_bands=5, epsilon=0.2, seed=42)
        
        for t in range(5):
            actionA = schedA.select_action()
            actionB = schedB.select_action()
            
            # Action should be identical
            self.assertEqual(actionA, actionB)
            
            detA = recA.scan(envA, actionA, dwell_time=1)
            detB = recB.scan(envB, actionB, dwell_time=1)
            
            # Observations should be identical
            self.assertEqual(detA, detB)
            
            schedA.update(None, actionA, detA)
            schedB.update(None, actionB, detB)
            
        # At this point, the beliefs and internal states must be identical
        stateA = schedA.get_state()
        stateB = schedB.get_state()
        self.assertTrue(np.allclose(stateA["beliefs"], stateB["beliefs"]))
        self.assertTrue(np.all(stateA["observations"] == stateB["observations"]))
        
        # Prove that hidden truth changing at t=5 didn't affect the next action taken at t=5
        self.assertEqual(schedA.select_action(), schedB.select_action())

if __name__ == "__main__":
    unittest.main()
